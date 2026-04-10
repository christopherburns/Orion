import Foundation
import MLX
import MLXNN

/// A node in the MCTS search tree.
///
/// Reference type so the tree can be mutated across simulations without copying.
/// Each node represents a game state reached via a unique path from the root.
final class MCTSNode {
   var visitCount: Int = 0
   var totalValue: Float = 0.0
   var priors: [Float] = []           // P(s,a) for all CANONICAL_MOVE_COUNT actions; empty until expanded
   var children: [Int: MCTSNode] = [:]

   var isExpanded: Bool { !priors.isEmpty }

   /// Mean backed-up value through action `a`, from THIS node's player's perspective.
   /// Child totalValue is stored from the child's player's perspective, so we negate
   /// to convert to the parent's perspective before using it in PUCT.
   func q (action: Int) -> Float {
      guard let child: MCTSNode = children[action], child.visitCount > 0 else { return 0.0 }
      return -(child.totalValue / Float(child.visitCount))
   }

   /// PUCT score for action `a`: balances exploitation (Q) vs exploration (prior + visit bonus).
   func puct (action: Int, cPuct: Float) -> Float {
      let childVisits = Float(children[action]?.visitCount ?? 0)
      let explorationBonus = cPuct * priors[action] * sqrt(Float(visitCount)) / (1.0 + childVisits)
      return q(action: action) + explorationBonus
   }
}

/// AlphaZero-style MCTS using a PolicyValueNetwork for both prior guidance and leaf evaluation.
///
/// Each call to `search()` runs `monteCarloSamples` simulations from the root position,
/// building a search tree that persists across simulations. The tree grows by one
/// leaf per simulation; depth accumulates naturally on the most-visited branches.
///
/// Returns a visit-count policy distribution π, which is used as the policy training
/// target — a richer signal than a one-hot because it captures relative move quality.
///
/// Note: Currently implements negamax value backup, which is correct for 2-player
/// zero-sum games. For 3-4 player support, per-player value tracking is needed.
public struct MCTSSearch {
   let network: PolicyValueNetwork
   public let monteCarloSamples: Int
   public let cPuct: Float
   public let debug: Bool

   init (network: PolicyValueNetwork, monteCarloSamples: Int, cPuct: Float, debug: Bool = false) {
      self.network = network
      self.monteCarloSamples = monteCarloSamples
      self.cPuct = cPuct
      self.debug = debug
   }

   /// Run MCTS from the current game state and return a policy distribution.
   ///
   /// - Parameters:
   ///   - game: Current game state. Not mutated — each simulation works on a copy.
   ///   - temperature: Controls sharpness of the returned distribution.
   ///                  τ=1 returns raw normalized visit counts (exploratory).
   ///                  τ→0 returns a one-hot on the most-visited action (greedy).
   /// - Returns: [Float] of length CANONICAL_MOVE_COUNT summing to 1.0 over legal moves.
   public func search (game: Game, temperature: Float) -> [Float] {
      network.train(false)  // inference mode — disable dropout
      let root = MCTSNode()

      for _ in 0..<monteCarloSamples {
         var gameCopy = game
         simulate(node: root, game: &gameCopy)
      }

      let policy = visitCountPolicy(root: root, temperature: temperature)

      if debug {
         printSearchResults(root: root, policy: policy, turn: game.currentTurn)
      }

      return policy
   }

   // MARK: - Simulation

   /// One simulation: selection → expansion/evaluation → backpropagation.
   ///
   /// Returns value from the perspective of the player to move at this node.
   /// Negates when the active player changes so Q values stay in the right frame.
   @discardableResult
   private func simulate (node: MCTSNode, game: inout Game) -> Float {
      let value: Float

      switch game.terminalCondition {
      case .playerWon(let playerIndex):
         // Value from the perspective of whoever is "current player" at this terminal state.
         // After a winning move is applied, currentPlayer has already advanced to the next
         // player — so the winner is NOT currentPlayer here. The negation in the caller
         // flips this back to the correct perspective for the parent node.
         value = (playerIndex == game.currentPlayer) ? 1.0 : -1.0

      case .tied, .timedOut:
         value = 0.0

      case .inProgress:
         if !node.isExpanded {
            // Leaf: run network to get priors + value estimate, then stop.
            value = expandAndEvaluate(node: node, game: game)
         } else {
            // Interior node: select best action by PUCT, recurse, backprop.
            value = selectAndDescend(node: node, game: &game)
         }
      }

      node.visitCount += 1
      node.totalValue += value
      return value
   }

   /// Select the best action by PUCT, apply it, recurse into the child.
   private func selectAndDescend (node: MCTSNode, game: inout Game) -> Float {
      let legalMask = game.legalMoveMaskForCurrentPlayer()
      var bestAction = -1
      var bestScore = -Float.infinity

      for action in 0..<Game.CANONICAL_MOVE_COUNT {
         guard legalMask[action] else { continue }
         let score = node.puct(action: action, cPuct: cPuct)
         if score > bestScore {
            bestScore = score
            bestAction = action
         }
      }
      precondition(bestAction >= 0, "MCTS: no legal action found in non-terminal position")

      if node.children[bestAction] == nil {
         node.children[bestAction] = MCTSNode()
      }
      let child = node.children[bestAction]!

      let playerBefore = game.currentPlayer
      game.applyMove(canonicalMoveIndex: bestAction)
      let playerAfter = game.currentPlayer

      var childValue = simulate(node: child, game: &game)

      // Negate when the active player changed — value flips perspective.
      // Stays the same during discard phase when the same player continues.
      if playerAfter != playerBefore {
         childValue = -childValue
      }

      return childValue
   }

   // MARK: - Expansion

   /// Expand a leaf node: run network to get policy priors and a value estimate.
   /// Stores masked-softmax priors so future simulations can select through this node.
   private func expandAndEvaluate (node: MCTSNode, game: Game) -> Float {
      let stateFloats = game.encoding().map { Float($0) }
      let stateArray = MLXArray(stateFloats).reshaped([1, PolicyValueNetwork.INPUT_DIMENSIONS])
      let (policyLogits, valueArr) = network.execute(stateArray)

      let logits = policyLogits[0].asArray(Float.self)
      let legalMask = game.legalMoveMaskForCurrentPlayer()
      node.priors = maskedSoftmax(logits: logits, mask: legalMask)

      return valueArr[0, 0].item(Float.self)
   }

   /// Softmax restricted to legal moves. Illegal moves get probability 0.
   private func maskedSoftmax (logits: [Float], mask: [Bool]) -> [Float] {
      var maxLogit = -Float.infinity
      for i in 0..<logits.count where mask[i] {
         maxLogit = max(maxLogit, logits[i])
      }

      var result = [Float](repeating: 0.0, count: logits.count)
      var sumExp: Float = 0.0
      for i in 0..<logits.count where mask[i] {
         let e = exp(logits[i] - maxLogit)
         result[i] = e
         sumExp += e
      }

      if sumExp > 0 {
         for i in 0..<result.count {
            result[i] /= sumExp
         }
      }
      return result
   }

   // MARK: - Policy extraction

   /// Convert root visit counts to a probability distribution with temperature.
   ///
   ///   π(a) ∝ N(root, a)^(1/τ)
   ///
   /// At τ=0 (greedy), returns a one-hot on the most-visited action.
   private func visitCountPolicy (root: MCTSNode, temperature: Float) -> [Float] {
      var counts = [Float](repeating: 0.0, count: Game.CANONICAL_MOVE_COUNT)
      for (action, child) in root.children {
         counts[action] = Float(child.visitCount)
      }

      if temperature < 1e-6 {
         var policy = [Float](repeating: 0.0, count: Game.CANONICAL_MOVE_COUNT)
         if let best = counts.indices.max(by: { counts[$0] < counts[$1] }) {
            policy[best] = 1.0
         }
         return policy
      }

      var powered = counts.map { pow($0, 1.0 / temperature) }
      let total = powered.reduce(0, +)
      if total > 0 {
         powered = powered.map { $0 / total }
      }
      return powered
   }

   // MARK: - Debug output

   /// Short human-readable name for a canonical move index.
   private static func moveName (_ action: Int) -> String {
      let gems = ["R", "G", "B", "W", "K"]   // red, green, blue, white, brown
      let allGems = ["R", "G", "B", "W", "K", "★"]  // includes gold for discard
      switch action {
      case  0..<4:  return "buy1[\(action)]"
      case  4..<8:  return "buy2[\(action - 4)]"
      case  8..<12: return "buy3[\(action - 8)]"
      case 12..<15: return "buyR[\(action - 12)]"
      case 15..<25: return "3gem[\(action - 15)]"
      case 25..<30: return "2gem[\(gems[action - 25])]"
      case 30..<42: return "rsv\((action - 30) / 4 + 1)[\((action - 30) % 4)]"
      case 42..<48: return "disc[\(allGems[action - 42])]"
      default:      return "mv[\(action)]"
      }
   }

   /// Print the search tree and π distribution after a completed search.
   private func printSearchResults (root: MCTSNode, policy: [Float], turn: Int) {
      print("\n┌─ MCTS  \(monteCarloSamples) sims · turn \(turn) · root N=\(root.visitCount)  (Q = current player's perspective at each level)")

      let depth1 = root.children.sorted { $0.value.visitCount > $1.value.visitCount }
      for (i, (action, child)) in depth1.enumerated() {
         let isLast = i == depth1.count - 1
         printNode(action: action, node: child,
                   prior: action < root.priors.count ? root.priors[action] : 0,
                   linePrefix:  isLast ? "└──" : "├──",
                   childPrefix: isLast ? "    " : "│   ",
                   depth: 1, maxDepth: 3, maxSiblings: 5)
      }

      // π bar chart — non-zero entries sorted by probability
      let nonZero = policy.enumerated()
         .filter { $0.element > 0.001 }
         .sorted { $0.element > $1.element }

      let maxP = nonZero.first?.element ?? 1.0
      let barWidth = 20
      print("\n  π  (\(nonZero.count) legal moves explored):")
      for (action, prob) in nonZero {
         let filled = Int((prob / maxP) * Float(barWidth))
         let bar = String(repeating: "█", count: filled)
               + String(repeating: "░", count: barWidth - filled)
         let name = Self.moveName(action)
         print("    \(name.padding(toLength: 10, withPad: " ", startingAt: 0))  \(bar)  \(String(format: "%.3f", prob))")
      }
      print("")
   }

   /// Recursively print one node of the search tree.
   private func printNode (action: Int, node: MCTSNode,
      prior: Float,
      linePrefix: String, childPrefix: String,
      depth: Int, maxDepth: Int, maxSiblings: Int) {
         
      let name   = Self.moveName(action)
      let q      = node.visitCount > 0 ? node.totalValue / Float(node.visitCount) : 0.0
      let nStr   = String(node.visitCount).padding(toLength: 4, withPad: " ", startingAt: 0)

      // Target column for "N=": linePrefix (grows 4/level) + 1 space + name + 2 separator.
      // nameWidth = targetCol - prefixLen - 1 - 2, clamped so we never truncate the name.
      let targetCol = 26
      let nameWidth = max(name.count, targetCol - linePrefix.count - 3)
      print("\(linePrefix) \(name.padding(toLength: nameWidth, withPad: " ", startingAt: 0))  N=\(nStr)  Q=\(String(format: "%+.3f", q))  P=\(String(format: "%.3f", prior))")

      guard depth < maxDepth, !node.children.isEmpty else { return }

      let children = node.children
         .sorted { $0.value.visitCount > $1.value.visitCount }
         .prefix(maxSiblings)

      let truncated = node.children.count > maxSiblings

      for (i, (childAction, childNode)) in children.enumerated() {
         let isLast = i == children.count - 1 && !truncated
         let childPrior = childAction < node.priors.count ? node.priors[childAction] : 0.0
         printNode(
            action: childAction, node: childNode,
            prior: childPrior,
            linePrefix:  childPrefix + (isLast ? "└──" : "├──"),
            childPrefix: childPrefix + (isLast ? "    " : "│   "),
            depth: depth + 1, maxDepth: maxDepth, maxSiblings: maxSiblings)
      }

      if truncated {
         print("\(childPrefix)└── … (\(node.children.count - maxSiblings) more)")
      }
   }
}
