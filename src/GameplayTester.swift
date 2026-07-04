import Swift
import Foundation
import Core
import MLX
import Splendor
import Utility

public struct GameplayTester {

   static func registerOptions (opts: OptionParser) {
      opts.addOption("Gameplay Tester", "s", "seed", "Seed for random number generator (default: 42)")
      opts.addOption("Gameplay Tester", "n", "game-count", "Number of games to play (default: 1)")
      opts.addOption("Gameplay Tester", "p", "player-count", "Number of players [2-4] (default: 2)")

      opts.addOption("Gameplay Tester", "a", "agent", "Path to model file, or name of non-model-based agent (optional, default is 'random')",
         longDoc:
            "Specifies which agent(s) to use for gameplay. " +
            "If a path to a model file is provided (e.g., 'models/best.mlx'), " +
            "a neural network agent will be loaded from that file. " +
            "Alternatively, you can specify 'random' to use a random agent " +
            "that makes valid moves uniformly at random (default). " +
            "The model file path can be relative to the current working " +
            "directory or an absolute path. The number of arguments must " +
            "match the number of players, or if one is supplied it will be " +
            "instanced for all players.")

      opts.addOption("Gameplay Tester", "t", "temperature", "Sampling temperature for move selection (0 = greedy, default: 0)")
      opts.addOption("Gameplay Tester", "v", "verbose", "Show detailed game output even for multiple games (default: auto)")
      opts.addOption("Gameplay Tester", "", "show-probabilities", "Show move probability distribution for each turn (default: false)")
      opts.addOption("Gameplay Tester", "", "max-turns", "Maximum turns per game before timeout (default: 1000)")
      opts.addOption("Gameplay Tester", "b", "batch-size", "Number of games to run in parallel (default: 64)")
      opts.addOption("Gameplay Tester", "", "serial", "Force single-threaded evaluation (default: concurrent)", requireArgument: false)
      opts.addOption("Gameplay Tester", "", "precision", "Numeric precision for the neural agent: fp32, bf16, fp16 (default: fp32)")
      opts.addOption("Gameplay Tester", "", "monte-carlo-samples", "MCTS simulations per move (0 = disabled, use raw policy prediction; default: 0)")
      opts.addOption("Gameplay Tester", "", "determinizations", "Independent hidden-deck-order samples searched per move and aggregated together — avoids the search exploiting deck order no real player could see (default: 8)",
         longDoc:
            "Splendor's only hidden information is the face-down portion of each " +
            "tier's deck: nobody, including this engine, knows what card is " +
            "under the visible row until it's drawn. Naively running MCTS " +
            "directly against the game's one true (but player-invisible) deck " +
            "order would let the search exploit knowledge no real opponent has. " +
            "Instead, each move samples this many independent random reshuffles " +
            "of the still-hidden cards (see Game.determinized(seed:)), runs a " +
            "full search on each, and aggregates visit counts across all of " +
            "them before picking a move. Only used when --monte-carlo-samples " +
            "is enabled; self-play data generation does not need this.")
      opts.addOption("Gameplay Tester", "", "mcts-leaf-batch", "Leaves selected per MCTS round via virtual loss, same as `orion generate` (default: 8)")
      opts.addOption("Gameplay Tester", "", "c-puct", "MCTS exploration constant (default: 1.5)")
      opts.addOption("Gameplay Tester", "", "report-json", "Write a structured JSON report of inputs and results to this path")
   }


   // MARK: - Structured report

   struct PlayReport: Encodable {
      let schemaVersion: Int
      let command: String
      let startedAt: String
      let completedAt: String
      let elapsedSeconds: Double
      let parameters: Parameters
      let results: Results

      struct AgentSpec: Encodable {
         let spec: String
         let kind: String
         let label: String
      }

      struct Parameters: Encodable {
         let agents: [AgentSpec]
         let gameCount: Int
         let playerCount: Int
         let temperature: Float
         let maxTurns: Int
         let seed: UInt64
         let batchSize: Int
         let monteCarloSamples: Int
         let determinizations: Int
         let cPuct: Float
         let mctsLeafBatch: Int
      }

      struct PerPlayerResult: Encodable {
         let index: Int
         let spec: String
         let kind: String
         let label: String
         let wins: Int
         let winRateOverAll: Double
         let winRateOverDecisive: Double
      }

      struct Results: Encodable {
         let totalGames: Int
         let decisiveGames: Int
         let tiedGames: Int
         let timedOutGames: Int
         let totalTurns: Int
         let avgTurnsPerGame: Double
         let perPlayer: [PerPlayerResult]
      }
   }

   /// Convert logits to probabilities by masking illegal moves and applying softmax
   /// - Parameters:
   ///   - logits: Raw network output logits for all moves
   ///   - validMoveMask: Boolean mask indicating which moves are legal
   /// - Returns: Probability distribution over all moves (illegal moves have probability 0), or nil if no legal moves
   static func computeMoveProbabilities (logits: [Float], validMoveMask: [Bool]) -> [Float]? {
      precondition(validMoveMask.count == logits.count, "Move mask and logits must have same length")

      // Mask illegal moves by setting their logits to -infinity
      var maskedLogits = logits
      for (index, isValid) in validMoveMask.enumerated() {
         if !isValid {
            maskedLogits[index] = -Float.infinity
         }
      }

      // Apply softmax to get probabilities
      // First find max for numerical stability
      let maxLogit = maskedLogits.max() ?? -Float.infinity
      guard maxLogit.isFinite else {
         // All moves are illegal
         return nil
      }

      // Compute exp(logit - maxLogit) for numerical stability
      let expScores = maskedLogits.map { exp($0 - maxLogit) }
      let sumExp = expScores.reduce(0.0, +)

      // Return normalized probabilities
      return expScores.map { $0 / sumExp }
   }

   /// Sample a move using greedy selection (argmax)
   /// - Parameters:
   ///   - validMoveMask: Boolean mask indicating which moves are legal
   ///   - movePreferences: Raw logits from the neural network
   /// - Returns: Index of the best move, or nil if no legal moves
   static func sampleMove (validMoveMask: [Bool], movePreferences: [Float]) -> Int? {
      guard let probabilities = computeMoveProbabilities(logits: movePreferences, validMoveMask: validMoveMask) else {
         return nil
      }

      // For now, use greedy selection (argmax)
      // TODO: Add temperature-based sampling for training
      return probabilities.enumerated().max(by: { $0.1 < $1.1 })?.0
   }

   static func showGameState (game: Splendor.Game) {
      GamePrinter.present(game)
      for (index, player) in game.players.enumerated() {
         GamePrinter.presentPlayer(player, playerIndex: index)
      }
   }

   static func playGame (playerCount: Int, silence: Bool, seed: UInt64, agents: [any AgentProtocol], temperature: Float = 0, maxTurns: Int = 1000) -> (GameTerminalCondition, Int) {

      precondition(agents.count == playerCount, "Number of agents must match player count")

      var rng = SeededRandomNumberGenerator(seed: seed)

      guard var g = Splendor.Game(playerCount: playerCount, seed: seed) else {
         print("Error: Failed to create game state")
         return (.inProgress, 0)
      }

      if !silence {
         showGameState(game: g)
      }

      // Track move statistics
      var moveTypeCounts: [String: Int] = [
         "purchase": 0,
         "purchaseReserved": 0,
         "takeThreeGems": 0,
         "takeTwoGems": 0,
         "reserve": 0,
         "discard": 0
      ]

      // Game loop
      var timedOut = false
      while case .inProgress = g.terminalCondition {
         if g.currentTurn >= maxTurns {
            timedOut = true
            break
         }

         let validMoveMask = g.legalMoveMaskForCurrentPlayer()
         let currentAgent = agents[g.currentPlayer]
         let (policyLogits, _) = currentAgent.predict(game: g, currentPlayerIndex: g.currentPlayer)

         let moveResult = temperature > 0
            ? sampleMoveWithTemperature(logits: policyLogits, validMoveMask: validMoveMask, temperature: temperature, rng: &rng)
            : sampleMove(validMoveMask: validMoveMask, movePreferences: policyLogits).map { ($0, [Float]()) }
         guard let (moveIndex, _) = moveResult else {
            print("Error: No valid moves available for player \(g.currentPlayer)")
            print("Valid move mask: \(validMoveMask)")
            print("All false? \(validMoveMask.allSatisfy { !$0 })")
            print("   Game phase: \(g.phase)")
            print("   Game state:")
            print("      Players:")
            GamePrinter.present(g)
            GamePrinter.presentPlayer(g.players[g.currentPlayer], playerIndex: g.currentPlayer)
            preconditionFailure("No valid moves available for player \(g.currentPlayer)")
            break
         }


         // Track move type
         if moveIndex < 12 {
            moveTypeCounts["purchase"]! += 1
         } else if moveIndex < 15 {
            moveTypeCounts["purchaseReserved"]! += 1
         } else if moveIndex < 25 {
            moveTypeCounts["takeThreeGems"]! += 1
         } else if moveIndex < 30 {
            moveTypeCounts["takeTwoGems"]! += 1
         } else if moveIndex < 42 {
            moveTypeCounts["reserve"]! += 1
         } else {
            moveTypeCounts["discard"]! += 1
         }

         g.applyMove(canonicalMoveIndex: moveIndex)

         if !silence {
            GamePrinter.presentMove(moveIndex: moveIndex, game: g)
            showGameState(game: g)
         }
      }

      // Print move statistics
      if !silence {
         print("\n\u{001B}[1mMove Statistics:\u{001B}[0m")
         for (moveType, count) in moveTypeCounts.sorted(by: { $0.key < $1.key }) {
            let percentage = Float(count) / Float(g.currentTurn) * 100.0
            print("  \(moveType): \(count) (\(String(format: "%.1f", percentage))%)")
         }
      }

      return (timedOut ? .timedOut : g.terminalCondition, g.currentTurn)
   }


   /// Interactive game loop for human vs CPU play. **2-player only.**
   /// Renders a quadrant layout each turn:
   ///   ┌─ Opponent ─┐  ┌─ Game Board ─┐
   ///   ┌─ You ──────┐  ┌─ Moves ──────┐
   /// Human player gets a numbered move list; CPU shows heat-colored probability bars.
   static func playGameInteractive (
      playerCount: Int,
      seed: UInt64,
      agents: [any AgentProtocol],
      maxTurns: Int = 1000) -> (GameTerminalCondition, Int) {

      precondition(agents.count == playerCount, "Number of agents must match player count")
      precondition(playerCount == 2, "Interactive mode supports 2-player games only")

      guard var g = Splendor.Game(playerCount: playerCount, seed: seed) else {
         print("Error: Failed to create game state")
         return (.inProgress, 0)
      }

      var timedOut = false

      while case .inProgress = g.terminalCondition {
         if g.currentTurn >= maxTurns {
            timedOut = true
            break
         }

         let currentAgent  = agents[g.currentPlayer]
         let validMoveMask = g.legalMoveMaskForCurrentPlayer()
         let legalIndices  = validMoveMask.indices.filter { validMoveMask[$0] }

         let moveIndex: Int

         if currentAgent.isHuman {
            // ── Human turn ────────────────────────────────────────────────
            GamePrinter.presentInteractive2P(game: g, legalMoveIndices: legalIndices)

            var chosen: Int? = nil
            while chosen == nil {
               print("\nEnter move (1–\(legalIndices.count)): ", terminator: "")
               fflush(stdout)
               if let line = readLine(), let n = Int(line.trimmingCharacters(in: .whitespaces)),
                  n >= 1 && n <= legalIndices.count {
                  chosen = legalIndices[n - 1]
               } else {
                  print("Invalid choice — enter a number between 1 and \(legalIndices.count).")
               }
            }
            moveIndex = chosen!

         } else {
            // ── CPU turn ──────────────────────────────────────────────────
            let (policyLogits, _) = currentAgent.predict(game: g, currentPlayerIndex: g.currentPlayer)

            // Greedy move selection
            guard let greedyIndex = sampleMove(validMoveMask: validMoveMask, movePreferences: policyLogits) else {
               preconditionFailure("CPU player \(g.currentPlayer) has no legal moves")
            }
            moveIndex = greedyIndex

            // Compute softmax probabilities over legal moves for display (temperature = 1)
            let probs = computeMoveProbabilities(logits: policyLogits, validMoveMask: validMoveMask)
               ?? Array(repeating: 0.0, count: policyLogits.count)

            GamePrinter.presentInteractive2P(
               game: g,
               legalMoveIndices: legalIndices,
               probabilities: probs,
               chosenIndex: moveIndex)

            print("\nPress enter to continue…", terminator: "")
            fflush(stdout)
            _ = readLine()
         }

         g.applyMove(canonicalMoveIndex: moveIndex)
      }

      // Final board state — use the legacy sequential render since the game is over
      // and there are no moves to show.
      print("\n" + String(repeating: "═", count: 80))
      GamePrinter.present(g)
      for (i, player) in g.players.enumerated() {
         GamePrinter.presentPlayer(player, playerIndex: i)
      }

      if timedOut {
         print("Game ended: turn limit (\(maxTurns)) reached.")
      } else if case .playerWon(let winner) = g.terminalCondition {
         let tag = agents[winner].isHuman ? "You win!" : "CPU wins."
         print("\(tag) Player \(winner + 1) reached \(g.players[winner].score) points in \(g.currentTurn) turns.")
      } else {
         print("Game tied after \(g.currentTurn) turns.")
      }

      return (timedOut ? .timedOut : g.terminalCondition, g.currentTurn)
   }


   /// Batched game evaluation — runs multiple games concurrently, grouping agent predictions by player.
   static func batchedPlayGames (
      gameCount: Int,
      playerCount: Int,
      seed: UInt64,
      agents: [any AgentProtocol],
      temperature: Float,
      maxTurns: Int,
      laneCount: Int,
      baseGameIndex: Int = 0) -> [(condition: GameTerminalCondition, turnCount: Int)] {

      struct EvalLane {
         var game: Splendor.Game
         var rng: SeededRandomNumberGenerator
         var gameIndex: Int
         var active: Bool
         // Turn count comes from game.currentTurn, which counts player turns
         // correctly (a discard sub-move does not advance it).
      }

      let actualLanes = min(laneCount, gameCount)
      var nextGameIndex = actualLanes

      func initLane (_ gameIndex: Int) -> EvalLane? {
         let gameSeed = seed + UInt64(gameIndex)
         guard let game = Splendor.Game(playerCount: playerCount, seed: gameSeed) else { return nil }
         return EvalLane(game: game, rng: SeededRandomNumberGenerator(seed: gameSeed), gameIndex: gameIndex, active: true)
      }

      var lanes = (0..<actualLanes).compactMap { initLane($0) }
      var results: [(condition: GameTerminalCondition, turnCount: Int)] = []

      // Prepare agents for inference
      for agent in agents { agent.prepareForInference() }

      while lanes.contains(where: { $0.active }) {
         // Group active lanes by current player index
         var groupsByPlayer: [Int: [Int]] = [:]  // playerIndex -> [lane indices]
         for i in lanes.indices where lanes[i].active {
            if lanes[i].game.currentTurn >= maxTurns {
               results.append((condition: .timedOut, turnCount: lanes[i].game.currentTurn))
               if nextGameIndex < gameCount, let lane = initLane(nextGameIndex) {
                  lanes[i] = lane; nextGameIndex += 1
               } else {
                  lanes[i].active = false
               }
               continue
            }
            let player = lanes[i].game.currentPlayer
            groupsByPlayer[player, default: []].append(i)
         }

         // Batch predict per agent
         for (playerIndex, laneIndices) in groupsByPlayer {
            let agent = agents[playerIndex]
            let games: [any GameProtocol] = laneIndices.map { lanes[$0].game }
            let playerIndices = laneIndices.map { lanes[$0].game.currentPlayer }
            let predictions = agent.batchPredict(games: games, currentPlayerIndices: playerIndices)

            for (j, laneIdx) in laneIndices.enumerated() {
               let logits = predictions[j].policyLogits
               let validMoveMask = lanes[laneIdx].game.legalMoveMaskForCurrentPlayer()

               let moveResult = temperature > 0
                  ? sampleMoveWithTemperature(logits: logits, validMoveMask: validMoveMask, temperature: temperature, rng: &lanes[laneIdx].rng)
                  : sampleMove(validMoveMask: validMoveMask, movePreferences: logits).map { ($0, [Float]()) }

               guard let (moveIndex, _) = moveResult else {
                  preconditionFailure("No valid moves in batched eval for game \(lanes[laneIdx].gameIndex)")
               }

               lanes[laneIdx].game.applyMove(canonicalMoveIndex: moveIndex)

               // Check terminal
               if case .inProgress = lanes[laneIdx].game.terminalCondition {} else {
                  results.append((condition: lanes[laneIdx].game.terminalCondition, turnCount: lanes[laneIdx].game.currentTurn))
                  if nextGameIndex < gameCount, let lane = initLane(nextGameIndex) {
                     lanes[laneIdx] = lane; nextGameIndex += 1
                  } else {
                     lanes[laneIdx].active = false
                  }
               }
            }
         }
      }

      return results
   }


   /// Batched game evaluation using determinized MCTS search instead of raw
   /// single-ply policy prediction — see Game.determinized(seed:) for why this
   /// is necessary. Splendor's only hidden information is each tier's face-down
   /// deck order, unknown to every player including this engine. Searching
   /// directly against the game's one true internal deck order would let the
   /// search see draws no real opponent could predict. Instead, each move
   /// independently reshuffles the hidden cards `determinizations` times, runs
   /// a full search on each reshuffle, and aggregates visit counts across all
   /// of them before picking a move — so the chosen move never depends on any
   /// single, player-invisible resolution of the unknown deck order.
   ///
   /// Self-play data generation (DataGenerator) does not use this: full
   /// information there is a fair, symmetric training signal (every game is
   /// bootstrapped from the same source of self-play), not a competitive-play
   /// advantage, so it is left untouched.
   ///
   /// Structurally this mirrors `batchedPlayGames` above — concurrent game
   /// lanes, replenished from a shared queue as each finishes — but every lane
   /// additionally carries `determinizations` parallel search trees for its
   /// current decision point. All lanes' pending leaf evaluations in a round
   /// are grouped by whichever player is to move at that leaf (which can differ
   /// from the lane's root player, since a tree can look several plies ahead)
   /// and routed to that player's own agent — matchups can mix different agents
   /// (e.g. champion vs. heuristic), unlike self-play's single shared agent.
   static func batchedPlayGamesWithMCTS (
      gameCount: Int,
      playerCount: Int,
      seed: UInt64,
      agents: [any AgentProtocol],
      temperature: Float,
      maxTurns: Int,
      laneCount: Int,
      monteCarloSamples: Int,
      determinizations: Int,
      cPuct: Float,
      mctsLeafBatch: Int,
      baseGameIndex: Int = 0) -> [(condition: GameTerminalCondition, turnCount: Int)] {

      precondition(determinizations >= 1, "determinizations must be at least 1")

      struct DeterminizationLane {
         var game: Splendor.Game        // true, observed state — never itself determinized
         var roots: [MCTSNode]          // one search tree per determinization
         var determinizedGames: [Splendor.Game]  // matching determinized copies, fixed for this decision
         var rng: SeededRandomNumberGenerator
         var gameIndex: Int
         var active: Bool
         // Turn count comes from game.currentTurn, which counts player turns
         // correctly (a discard sub-move does not advance it).
      }

      let actualLanes = min(laneCount, gameCount)
      var nextGameIndex = actualLanes
      var results: [(condition: GameTerminalCondition, turnCount: Int)] = []

      // Seed for determinization draws, advanced independently of any game's own
      // seed so reshuffles don't correlate with (and can't leak) the true order.
      var nextDeterminizationSeed: UInt64 = seed &+ 0x9E37_79B9_7F4A_7C15

      // DumbAgent's predict() returns a coin-flip value estimate (see
      // RandomAgent.swift) — it's fine as a raw single-ply policy (its logits
      // are legitimately uniform), but MCTS's exploitation term would treat
      // that coin-flip as a real skill signal and chase whichever branch
      // happened to get lucky early backprops, corrupting "random" into
      // something that isn't actually uniform. So DumbAgent's moves always
      // skip search — see the raw-sampling branch in the move-application
      // phase below — and this returns empty (no trees to build).
      func freshDeterminizations (for game: Splendor.Game) -> (roots: [MCTSNode], detGames: [Splendor.Game]) {
         guard !(agents[game.currentPlayer] is DumbAgent) else { return ([], []) }

         var games: [Splendor.Game] = []
         games.reserveCapacity(determinizations)
         for _ in 0..<determinizations {
            games.append(game.determinized(seed: nextDeterminizationSeed))
            nextDeterminizationSeed &+= 1
         }
         let roots = (0..<determinizations).map { _ in MCTSNode() }
         return (roots, games)
      }

      func initLane (_ gameIndex: Int) -> DeterminizationLane? {
         let globalIndex = baseGameIndex + gameIndex
         let gameSeed = seed + UInt64(gameIndex)
         guard let game = Splendor.Game(playerCount: playerCount, seed: gameSeed) else { return nil }
         let (roots, detGames) = freshDeterminizations(for: game)
         return DeterminizationLane(
            game: game, roots: roots, determinizedGames: detGames,
            rng: SeededRandomNumberGenerator(seed: gameSeed), gameIndex: globalIndex, active: true)
      }

      var lanes = (0..<actualLanes).compactMap { initLane($0) }

      for agent in agents { agent.prepareForInference() }

      // Only used for its agent-agnostic tree-walk/backprop methods
      // (selectLeavesWithVirtualLoss, completeEvaluation) — network calls are
      // routed to each leaf's own player's agent manually below, since a
      // matchup can involve more than one agent. `agents[0]` here is never
      // actually queried.
      let mctsHelper = MCTSSearch(agent: agents[0], monteCarloSamples: monteCarloSamples, cPuct: cPuct)
      let leafBatch = max(1, mctsLeafBatch)
      let roundCount = (monteCarloSamples + leafBatch - 1) / leafBatch

      while lanes.contains(where: { $0.active }) {
         // --- Simulation phase: batched selection + eval across every lane's
         // determinizations, grouped by whichever player is to move at each leaf.
         for _ in 0..<roundCount {
            var pendingByPlayer: [Int: [(laneIdx: Int, result: SelectionResult)]] = [:]

            for i in lanes.indices where lanes[i].active {
               // Empty roots means the current mover is a DumbAgent this
               // decision — no tree to search, handled by raw sampling below.
               for d in lanes[i].roots.indices {
                  let leafResults = mctsHelper.selectLeavesWithVirtualLoss(
                     root: lanes[i].roots[d], game: lanes[i].determinizedGames[d], count: leafBatch)
                  for result in leafResults {
                     let player = result.leafGame.currentPlayer
                     pendingByPlayer[player, default: []].append((i, result))
                  }
               }
            }

            for (player, group) in pendingByPlayer {
               let leafGames: [any GameProtocol] = group.map { $0.result.leafGame }
               let playerIndices = group.map { $0.result.leafGame.currentPlayer }
               let predictions = agents[player].batchPredict(games: leafGames, currentPlayerIndices: playerIndices)
               for (j, entry) in group.enumerated() {
                  mctsHelper.completeEvaluation(result: entry.result, logits: predictions[j].policyLogits, value: predictions[j].valueEstimate)
               }
            }
         }

         // --- Move application phase ---
         for i in lanes.indices where lanes[i].active {
            if lanes[i].game.currentTurn >= maxTurns {
               results.append((condition: .timedOut, turnCount: lanes[i].game.currentTurn))
               if nextGameIndex < gameCount, let lane = initLane(nextGameIndex) {
                  lanes[i] = lane; nextGameIndex += 1
               } else {
                  lanes[i].active = false
               }
               continue
            }

            let moveIndex: Int
            if lanes[i].roots.isEmpty {
               // DumbAgent's turn — search would corrupt its behavior (see
               // freshDeterminizations above), so fall back to the same raw
               // temperature-based sampling batchedPlayGames uses.
               let player = lanes[i].game.currentPlayer
               let (logits, _) = agents[player].predict(game: lanes[i].game, currentPlayerIndex: player)
               let validMoveMask = lanes[i].game.legalMoveMaskForCurrentPlayer()
               let moveResult = temperature > 0
                  ? sampleMoveWithTemperature(logits: logits, validMoveMask: validMoveMask, temperature: temperature, rng: &lanes[i].rng)
                  : sampleMove(validMoveMask: validMoveMask, movePreferences: logits).map { ($0, [Float]()) }
               guard let (idx, _) = moveResult else {
                  preconditionFailure("No valid moves in determinized eval for game \(lanes[i].gameIndex)")
               }
               moveIndex = idx
            } 
            else {
               // Temperature is handled inside aggregatedPolicy (temp≈0
               // collapses to a one-hot on the most-visited action), so no
               // separate greedy branch is needed here.
               let policy = MCTSSearch.aggregatedPolicy(roots: lanes[i].roots, temperature: temperature)
               moveIndex = sampleMoveFromPolicy(policy, rng: &lanes[i].rng)
            }

            lanes[i].game.applyMove(canonicalMoveIndex: moveIndex)

            if case .inProgress = lanes[i].game.terminalCondition {
               // Same game continues — re-determinize for the next decision point.
               let (roots, detGames) = freshDeterminizations(for: lanes[i].game)
               lanes[i].roots = roots
               lanes[i].determinizedGames = detGames
            }
            else {
               results.append((condition: lanes[i].game.terminalCondition, turnCount: lanes[i].game.currentTurn))
               if nextGameIndex < gameCount, let lane = initLane(nextGameIndex) {
                  lanes[i] = lane; nextGameIndex += 1
               } 
               else {
                  lanes[i].active = false
               }
            }
         }
      }

      return results
   }


   public static func main () throws {
      let opts = OptionParser(help: "Play Splendor games using neural network or random agents")
      self.registerOptions(opts: opts)
      opts.parse(tokens: CommandLine.arguments, failOnUnknownOption: true, ignoreHelp: false)

      let playerCount = opts.get(option: "player-count", orElse: 2)
      let gameCount = opts.get(option: "game-count", orElse: 1)
      let seed = opts.get(option: "seed", orElse: UInt64(42))
      let agentSpecs = opts.getAll(option: "agent", as: String.self)
      let temperature = opts.get(option: "temperature", orElse: Float(0))
      let maxTurns = opts.get(option: "max-turns", orElse: 1000)
      let batchSize = opts.get(option: "batch-size", orElse: 64)
      let serial = opts.wasProvided(option: "serial")
      let precisionStr = opts.get(option: "precision", orElse: "fp32")
      let precision = parsePrecision(precisionStr)
      let monteCarloSamples = opts.get(option: "monte-carlo-samples", orElse: 0)
      let determinizations = opts.get(option: "determinizations", orElse: 8)
      let cPuct = opts.get(option: "c-puct", orElse: Float(1.5))
      let mctsLeafBatch = opts.get(option: "mcts-leaf-batch", orElse: 8)
      let reportPath: String? = opts.get(option: "report-json")
      let startDate = Date()

      // Print configuration
      let agentDesc = agentSpecs.isEmpty ? "random" : agentSpecs.joined(separator: ", ")
      print("Configuration:")
      print("  Games:            \(gameCount)")
      print("  Players per game: \(playerCount)")
      print("  Agent:            \(agentDesc)")
      print("  Temperature:      \(String(format: "%.2f", temperature))")
      print("  Max turns:        \(maxTurns)")
      print("  Seed:             \(seed)")
      if monteCarloSamples > 0 {
         print("  MCTS sims/move:   \(monteCarloSamples)  (c_puct=\(cPuct), leaf batch=\(mctsLeafBatch))")
         print("  Determinizations: \(determinizations)  (independent hidden-deck reshuffles per move, aggregated)")
      }

      // Initialize agents based on command-line specifications
      let agents = initializeAgents(playerCount: playerCount, agentSpecs: agentSpecs, seed: seed, precision: precision)

      // Interactive mode: any human player triggers a single interactive game
      if agents.contains(where: { $0.isHuman }) {
         if gameCount > 1 {
            print("Note: interactive mode — game count overridden to 1.")
         }
         _ = playGameInteractive(playerCount: playerCount, seed: seed, agents: agents, maxTurns: maxTurns)
         return
      }

      // Partition games into GCD tasks
      let taskCount = (gameCount + batchSize - 1) / batchSize
      print("  Batch size:       \(batchSize)")
      print("  Tasks:            \(taskCount)\(serial ? " (serial)" : " (concurrent)")")

      let workQueue: DispatchQueue
      if serial {
         workQueue = DispatchQueue(label: "orion.play.work")
      } else {
         workQueue = DispatchQueue(label: "orion.play.work", attributes: .concurrent)
      }
      let resultQueue = DispatchQueue(label: "orion.play.results")
      let group = DispatchGroup()

      let taskResults = UnsafeMutableBufferPointer<[(condition: GameTerminalCondition, turnCount: Int)]?>.allocate(capacity: taskCount)
      taskResults.initialize(repeating: nil)
      defer { taskResults.deallocate() }

      for taskIndex in 0..<taskCount {
         let taskOffset = taskIndex * batchSize
         let taskGameCount = min(batchSize, gameCount - taskOffset)
         let taskBaseSeed = seed + UInt64(taskOffset)

         group.enter()
         workQueue.async {
            let taskAgents = initializeAgents(playerCount: playerCount, agentSpecs: agentSpecs, seed: taskBaseSeed, precision: precision)

            let results: [(condition: GameTerminalCondition, turnCount: Int)]
            if monteCarloSamples > 0 {
               results = batchedPlayGamesWithMCTS(
                  gameCount: taskGameCount,
                  playerCount: playerCount,
                  seed: taskBaseSeed,
                  agents: taskAgents,
                  temperature: temperature,
                  maxTurns: maxTurns,
                  laneCount: taskGameCount,
                  monteCarloSamples: monteCarloSamples,
                  determinizations: determinizations,
                  cPuct: cPuct,
                  mctsLeafBatch: mctsLeafBatch,
                  baseGameIndex: taskOffset)
            } else {
               results = batchedPlayGames(
                  gameCount: taskGameCount,
                  playerCount: playerCount,
                  seed: taskBaseSeed,
                  agents: taskAgents,
                  temperature: temperature,
                  maxTurns: maxTurns,
                  laneCount: taskGameCount,
                  baseGameIndex: taskOffset)
            }

            taskResults[taskIndex] = results
            resultQueue.async {
               let completed = taskResults.compactMap({ $0 }).reduce(0, { $0 + $1.count })
               print("Completed \(completed)/\(gameCount) games...")
               group.leave()
            }
         }
      }

      group.wait()

      // Merge results from all tasks
      var gameResults: [(condition: GameTerminalCondition, turnCount: Int)] = []
      for taskIndex in 0..<taskCount {
         if let results = taskResults[taskIndex] {
            gameResults.append(contentsOf: results)
         }
      }

      var totalTurnCount = 0
      var playerWinCounts: [Int: Int] = [:]
      var tiedCount = 0
      var timedOutCount = 0
      for (condition, turnCount) in gameResults {
         totalTurnCount += turnCount
         switch condition {
         case .playerWon(let playerIndex):
            playerWinCounts[playerIndex, default: 0] += 1
         case .tied:
            tiedCount += 1
         case .timedOut:
            timedOutCount += 1
         case .inProgress:
            break
         }
      }

      let concludedCount = gameCount - timedOutCount
      print("Total turns: \(totalTurnCount)")
      print("Average turns/game: \(String(format: "%.1f", Float(totalTurnCount)/Float(gameCount)))")

      for (index, count) in playerWinCounts.sorted(by: { $0.key < $1.key }) {
         let spec = index < agentSpecs.count ? agentSpecs[index] : "random"
         print("Player \(index) (\(spec)) won \(count) games (\(String(format: "%.1f", Float(count) / Float(concludedCount) * 100.0))%)")
      }
      print("Tied games: \(tiedCount)")
      if timedOutCount > 0 {
         print("Timed out:  \(timedOutCount) games (exceeded \(maxTurns) turns)")
      }

      if let reportPath = reportPath {
         let endDate = Date()
         let decisiveCount = gameCount - tiedCount - timedOutCount
         let resolvedSpecs: [String] = (0..<playerCount).map { i in
            i < agentSpecs.count ? agentSpecs[i] : "random"
         }
         let agentSpecsForReport: [PlayReport.AgentSpec] = resolvedSpecs.map { s in
            PlayReport.AgentSpec(
               spec: s, kind: Report.agentKind(spec: s), label: Report.agentLabel(spec: s))
         }
         let perPlayer: [PlayReport.PerPlayerResult] = (0..<playerCount).map { i in
            let s = resolvedSpecs[i]
            let wins = playerWinCounts[i] ?? 0
            return PlayReport.PerPlayerResult(
               index: i,
               spec:  s,
               kind:  Report.agentKind(spec: s),
               label: Report.agentLabel(spec: s),
               wins:  wins,
               winRateOverAll:      gameCount > 0     ? Double(wins) / Double(gameCount)     : 0,
               winRateOverDecisive: decisiveCount > 0 ? Double(wins) / Double(decisiveCount) : 0)
         }
         let report = PlayReport(
            schemaVersion:  Report.SCHEMA_VERSION,
            command:        "play",
            startedAt:      Report.timestamp(startDate),
            completedAt:    Report.timestamp(endDate),
            elapsedSeconds: endDate.timeIntervalSince(startDate),
            parameters: PlayReport.Parameters(
               agents:            agentSpecsForReport,
               gameCount:         gameCount,
               playerCount:       playerCount,
               temperature:       temperature,
               maxTurns:          maxTurns,
               seed:              seed,
               batchSize:         batchSize,
               monteCarloSamples: monteCarloSamples,
               determinizations:  determinizations,
               cPuct:             cPuct,
               mctsLeafBatch:     mctsLeafBatch),
            results: PlayReport.Results(
               totalGames:      gameCount,
               decisiveGames:   decisiveCount,
               tiedGames:       tiedCount,
               timedOutGames:   timedOutCount,
               totalTurns:      totalTurnCount,
               avgTurnsPerGame: gameCount > 0 ? Double(totalTurnCount) / Double(gameCount) : 0,
               perPlayer:       perPlayer))
         try Report.write(report, to: reportPath)
         print("Report written to: \(reportPath)")
      }
   }
}
