import Foundation
import Core

// MARK: - Batching support types

/// Records one step on the MCTS selection path for backpropagation.
public struct PathStep {
   let node: MCTSNode
   let didChangePlayer: Bool
}

/// Result of iterative MCTS leaf selection — either a leaf needing network eval, or a terminal node.
public struct SelectionResult {
   public let leafNode: MCTSNode
   public let path: [PathStep]
   public let leafGame: Game
   /// Non-nil when the leaf is a terminal state; backprop without calling the network.
   public let terminalValue: Float?
}

// MARK: - MCTSNode

/// A node in the MCTS search tree.
///
/// Reference type so the tree can be mutated in-place across simulations.
/// `children` is an empty array until this node is expanded; after expansion
/// it is a CANONICAL_MOVE_COUNT-slot array indexed directly by action.
public final class MCTSNode {
   var visitCount: Int = 0
   var totalValue: Float = 0.0
   var priors: [Float] = []
   var children: [MCTSNode?] = []

   public init () {}

   var isExpanded: Bool { !priors.isEmpty }

   /// Q value for `action` from THIS node's player's perspective.
   /// Child stores from the child's perspective, so we negate.
   func q (action: Int) -> Float {
      guard !children.isEmpty, let child = children[action], child.visitCount > 0 else { return 0.0 }
      return -(child.totalValue / Float(child.visitCount))
   }

   /// PUCT score for action `a`: balances exploitation (Q) vs exploration (prior + visit bonus).
   func puct (action: Int, cPuct: Float) -> Float {
      let childVisits = Float(children.isEmpty ? 0 : (children[action]?.visitCount ?? 0))
      let explorationBonus = cPuct * priors[action] * sqrt(Float(visitCount)) / (1.0 + childVisits)
      return q(action: action) + explorationBonus
   }
}

// MARK: - MCTSSearch

/// AlphaZero-style MCTS using any AgentProtocol for prior guidance and leaf evaluation.
///
/// Provides two usage modes:
/// 1. `search()` — single-game, self-contained (used by GameplayTester/evaluation)
/// 2. Batching primitives (`selectLeaf`, `batchEvaluate`, `expandLeaf`, `backpropagate`) —
///    used by DataGenerator to run many games in parallel.
public struct MCTSSearch {
   public let agent: any AgentProtocol
   public let monteCarloSamples: Int
   public let cPuct: Float
   public let debug: Bool

   public init (agent: any AgentProtocol, monteCarloSamples: Int, cPuct: Float, debug: Bool = false) {
      self.agent = agent
      self.monteCarloSamples = monteCarloSamples
      self.cPuct = cPuct
      self.debug = debug
   }

   // MARK: - Single-game search (used by GameplayTester)

   /// Run MCTS from the current game state and return a policy distribution.
   public func search (game: Game, temperature: Float) -> [Float] {
      let root = MCTSNode()

      for _ in 0..<monteCarloSamples {
         let result = selectLeaf(root: root, game: game)
         if let terminalValue = result.terminalValue {
            backpropagate(result: result, leafValue: terminalValue)
         } else {
            let (logits, value) = agent.predict(game: result.leafGame, currentPlayerIndex: result.leafGame.currentPlayer)
            let legalMask = result.leafGame.legalMoveMaskForCurrentPlayer()
            expandLeaf(node: result.leafNode, logits: logits, legalMask: legalMask)
            backpropagate(result: result, leafValue: value)
         }
      }

      let policy = visitCountPolicy(root: root, temperature: temperature)
      if debug { printSearchResults(root: root, policy: policy, turn: game.currentTurn) }
      return policy
   }

   // MARK: - Batching primitives

   /// Descend the tree from `root` to an unexpanded leaf or terminal node.
   /// Pure selection via PUCT — no network calls.
   public func selectLeaf (root: MCTSNode, game: Game) -> SelectionResult {
      var node = root
      var gameCopy = game
      var path: [PathStep] = []

      while true {
         switch gameCopy.terminalCondition {
         case .playerWon(let playerIndex):
            let value: Float = (playerIndex == gameCopy.currentPlayer) ? 1.0 : -1.0
            return SelectionResult(leafNode: node, path: path, leafGame: gameCopy, terminalValue: value)
         case .tied, .timedOut:
            return SelectionResult(leafNode: node, path: path, leafGame: gameCopy, terminalValue: 0.0)
         case .inProgress:
            if !node.isExpanded {
               return SelectionResult(leafNode: node, path: path, leafGame: gameCopy, terminalValue: nil)
            }
            let legalMask = gameCopy.legalMoveMaskForCurrentPlayer()
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
            let playerBefore = gameCopy.currentPlayer
            gameCopy.applyMove(canonicalMoveIndex: bestAction)
            let playerAfter = gameCopy.currentPlayer

            path.append(PathStep(node: node, didChangePlayer: playerAfter != playerBefore))
            node = node.children[bestAction]!
         }
      }
   }

   /// Set priors on an unexpanded leaf node from pre-computed network output.
   public func expandLeaf (node: MCTSNode, logits: [Float], legalMask: [Bool]) {
      node.priors = maskedSoftmax(logits: logits, mask: legalMask)
      node.children = [MCTSNode?](repeating: nil, count: Game.CANONICAL_MOVE_COUNT)
   }

   /// Walk the selection path in reverse, updating visit counts and backed-up values.
   public func backpropagate (result: SelectionResult, leafValue: Float) {
      result.leafNode.visitCount += 1
      result.leafNode.totalValue += leafValue

      var value = leafValue
      for step in result.path.reversed() {
         if step.didChangePlayer { value = -value }
         step.node.visitCount += 1
         step.node.totalValue += value
      }
   }

   /// Evaluate multiple leaf game states via batched agent prediction.
   public func batchEvaluate (leafGames: [Game]) -> (policyLogits: [[Float]], values: [Float]) {
      let games: [any GameProtocol] = leafGames
      let playerIndices = leafGames.map { $0.currentPlayer }
      let results = agent.batchPredict(games: games, currentPlayerIndices: playerIndices)

      var policyLogits: [[Float]] = []
      var values: [Float] = []
      policyLogits.reserveCapacity(results.count)
      values.reserveCapacity(results.count)
      for r in results {
         policyLogits.append(r.policyLogits)
         values.append(r.valueEstimate)
      }
      return (policyLogits, values)
   }

   // MARK: - Policy extraction

   /// Convert root visit counts to a probability distribution with temperature.
   ///   pi(a) = N(root, a)^(1/tau)
   /// At tau=0 (greedy), returns a one-hot on the most-visited action.
   public func visitCountPolicy (root: MCTSNode, temperature: Float) -> [Float] {
      var counts = [Float](repeating: 0.0, count: Game.CANONICAL_MOVE_COUNT)
      for (action, child) in root.children.enumerated() {
         if let child = child {
            counts[action] = Float(child.visitCount)
         }
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
      if total > 0 { powered = powered.map { $0 / total } }
      return powered
   }

   // MARK: - Helpers

   /// Softmax restricted to legal moves. Illegal moves get probability 0.
   private func maskedSoftmax (logits: [Float], mask: [Bool]) -> [Float] {
      var maxLogit = -Float.infinity
      for i in 0..<logits.count where mask[i] { maxLogit = max(maxLogit, logits[i]) }

      var result = [Float](repeating: 0.0, count: logits.count)
      var sumExp: Float = 0.0
      for i in 0..<logits.count where mask[i] {
         let e = exp(logits[i] - maxLogit)
         result[i] = e
         sumExp += e
      }
      if sumExp > 0 { for i in 0..<result.count { result[i] /= sumExp } }
      return result
   }

   // MARK: - Debug output

   /// Short human-readable name for a canonical move index.
   private static func moveName (_ action: Int) -> String {
      let gems = ["R", "G", "B", "W", "K"]
      let allGems = ["R", "G", "B", "W", "K", "★"]
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

   /// Print the search tree and policy distribution after a completed search.
   public func printSearchResults (root: MCTSNode, policy: [Float], turn: Int) {
      print("\n┌─ MCTS  \(monteCarloSamples) sims · turn \(turn) · root N=\(root.visitCount)  (Q = current player's perspective at each level)")

      let depth1 = root.children.enumerated()
         .compactMap { action, child in child.map { (action, $0) } }
         .sorted { $0.1.visitCount > $1.1.visitCount }

      for (i, (action, child)) in depth1.enumerated() {
         let isLast = i == depth1.count - 1
         printNode(action: action, node: child,
                   prior: action < root.priors.count ? root.priors[action] : 0,
                   linePrefix:  isLast ? "└──" : "├──",
                   childPrefix: isLast ? "    " : "│   ",
                   depth: 1, maxDepth: 3, maxSiblings: 5)
      }

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

      let targetCol = 26
      let nameWidth = max(name.count, targetCol - linePrefix.count - 3)
      print("\(linePrefix) \(name.padding(toLength: nameWidth, withPad: " ", startingAt: 0))  N=\(nStr)  Q=\(String(format: "%+.3f", q))  P=\(String(format: "%.3f", prior))")

      guard depth < maxDepth, !node.children.isEmpty else { return }

      let childPairs = node.children.enumerated()
         .compactMap { action, child in child.map { (action, $0) } }
         .sorted { $0.1.visitCount > $1.1.visitCount }

      let siblings = childPairs.prefix(maxSiblings)
      let truncated = childPairs.count > maxSiblings

      for (i, (childAction, childNode)) in siblings.enumerated() {
         let isLast = i == siblings.count - 1 && !truncated
         let childPrior = childAction < node.priors.count ? node.priors[childAction] : 0.0
         printNode(
            action: childAction, node: childNode,
            prior: childPrior,
            linePrefix:  childPrefix + (isLast ? "└──" : "├──"),
            childPrefix: childPrefix + (isLast ? "    " : "│   "),
            depth: depth + 1, maxDepth: maxDepth, maxSiblings: maxSiblings)
      }

      if truncated {
         print("\(childPrefix)└── … (\(childPairs.count - maxSiblings) more)")
      }
   }
}
