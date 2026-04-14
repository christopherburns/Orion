import Swift
import Foundation
import Core
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


   /// Interactive game loop for human vs CPU play.
   /// Prints board + player state before every turn. Human players get a numbered
   /// move menu; CPU players see heat-colored probability bars with the chosen move
   /// highlighted and a card panel when applicable.
   static func playGameInteractive (
      playerCount: Int,
      seed: UInt64,
      agents: [any AgentProtocol],
      maxTurns: Int = 1000) -> (GameTerminalCondition, Int) {

      precondition(agents.count == playerCount, "Number of agents must match player count")

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

         // Print current board and all player states
         GamePrinter.present(g)
         for (i, player) in g.players.enumerated() {
            GamePrinter.presentPlayer(player, playerIndex: i)
         }

         let currentAgent  = agents[g.currentPlayer]
         let validMoveMask = g.legalMoveMaskForCurrentPlayer()
         let legalIndices  = validMoveMask.indices.filter { validMoveMask[$0] }

         let moveIndex: Int

         if currentAgent.isHuman {
            // ── Human turn ────────────────────────────────────────────────
            GamePrinter.presentHumanMoveMenu(
               playerIndex: g.currentPlayer,
               legalMoveIndices: legalIndices,
               game: g)

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

            // Compute softmax probabilities over ALL moves for display (temperature = 1)
            let probs = computeMoveProbabilities(logits: policyLogits, validMoveMask: validMoveMask)
               ?? Array(repeating: 0.0, count: policyLogits.count)

            GamePrinter.presentCPUMoveMenu(
               playerIndex: g.currentPlayer,
               legalMoveIndices: legalIndices,
               probabilities: probs,
               chosenIndex: moveIndex,
               game: g)

            print("\nPress enter to continue…", terminator: "")
            fflush(stdout)
            _ = readLine()
         }

         g.applyMove(canonicalMoveIndex: moveIndex)
      }

      // Print final board state
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
         var turnCount: Int
         var gameIndex: Int
         var active: Bool
      }

      let actualLanes = min(laneCount, gameCount)
      var nextGameIndex = actualLanes

      func initLane (_ gameIndex: Int) -> EvalLane? {
         let gameSeed = seed + UInt64(gameIndex)
         guard let game = Splendor.Game(playerCount: playerCount, seed: gameSeed) else { return nil }
         return EvalLane(game: game, rng: SeededRandomNumberGenerator(seed: gameSeed), turnCount: 0, gameIndex: gameIndex, active: true)
      }

      var lanes = (0..<actualLanes).compactMap { initLane($0) }
      var results: [(condition: GameTerminalCondition, turnCount: Int)] = []

      // Prepare agents for inference
      for agent in agents { agent.prepareForInference() }

      while lanes.contains(where: { $0.active }) {
         // Group active lanes by current player index
         var groupsByPlayer: [Int: [Int]] = [:]  // playerIndex -> [lane indices]
         for i in lanes.indices where lanes[i].active {
            if lanes[i].turnCount >= maxTurns {
               results.append((condition: .timedOut, turnCount: lanes[i].turnCount))
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
               lanes[laneIdx].turnCount += 1

               // Check terminal
               if case .inProgress = lanes[laneIdx].game.terminalCondition {} else {
                  results.append((condition: lanes[laneIdx].game.terminalCondition, turnCount: lanes[laneIdx].turnCount))
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

      // Print configuration
      let agentDesc = agentSpecs.isEmpty ? "random" : agentSpecs.joined(separator: ", ")
      print("Configuration:")
      print("  Games:            \(gameCount)")
      print("  Players per game: \(playerCount)")
      print("  Agent:            \(agentDesc)")
      print("  Temperature:      \(String(format: "%.2f", temperature))")
      print("  Max turns:        \(maxTurns)")
      print("  Seed:             \(seed)")

      // Initialize agents based on command-line specifications
      let agents = initializeAgents(playerCount: playerCount, agentSpecs: agentSpecs, seed: seed)

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
            let taskAgents = initializeAgents(playerCount: playerCount, agentSpecs: agentSpecs, seed: taskBaseSeed)

            let results = batchedPlayGames(
               gameCount: taskGameCount,
               playerCount: playerCount,
               seed: taskBaseSeed,
               agents: taskAgents,
               temperature: temperature,
               maxTurns: maxTurns,
               laneCount: taskGameCount,
               baseGameIndex: taskOffset)

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
   }
}
