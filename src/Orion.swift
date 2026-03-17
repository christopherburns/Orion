
import Swift
import Foundation
import Core
import Splendor
import Utility

// MARK: - Orion CLI Tools
//
// Orion provides three main tools for training and testing neural networks on Splendor:
//
// 1. GENERATE - Generate training data via self-play
//    Usage: orion generate [options]
//    Example: orion generate -g 100 -m models/best.mlx -o training_data/run1.json -t 1.2
//
// 2. TRAIN - Train network on existing training data
//    Usage: orion train [options]
//    Example: orion train -i training_data/ -m models/checkpoint.mlx -o models/improved.mlx -b 256 -e 20
//
// 3. PLAY - Play games for testing and evaluation
//    Usage: orion play [options]
//    Example: orion play -n 100 -m models/best.mlx -m2 models/random
//
// Use --help with any tool to see available options.

@main
struct Orion {
   static func main () throws {

      if CommandLine.arguments.count == 1 {
         print("Usage: orion <command> [options]")
         print("Commands:")
         print("  train - Train a neural network model")
         print("  play - Play games using a neural network model")
         print("  generate - Generate training data for a neural network model")

         return
      }

      if CommandLine.arguments[1] == "train" {
         try NetworkTrainer.main()
      }
      else if CommandLine.arguments[1] == "play" {
         try GameplayTester.main()
      }
      else if CommandLine.arguments[1] == "generate" {
         try DataGenerator.main()
      }
      else {
         print("Unknown command: \(CommandLine.arguments[1])")
      }
   }
}
