import Swift
import Foundation
import Core
import Splendor
import Utility

public struct NetworkTrainer {

   static func registerOptions (opts: OptionParser) {
      opts.addOption("Network Trainer", "i", "input", "Input training data file or directory (required)")
      opts.addOption("Network Trainer", "m", "model", "Path to input model file to continue training (default: create new untrained model)")
      opts.addOption("Network Trainer", "o", "output", "Output path for trained model (default: models/model_TIMESTAMP.mlx)")
      opts.addOption("Network Trainer", "s", "seed", "Random seed for reproducibility (default: random)")
      opts.addOption("Network Trainer", "b", "batch-size", "Batch size for training (default: 256)")
      opts.addOption("Network Trainer", "e", "epochs", "Number of training epochs (default: 10)")
      opts.addOption("Network Trainer", "lr", "learning-rate", "Learning rate (default: 0.001)")
      opts.addOption("Network Trainer", "v", "validation-split", "Fraction of data to use for validation (default: 0.1)")
      opts.addOption("Network Trainer", "save-interval", "save-interval", "Save checkpoint every N epochs (default: 5)")
      opts.addOption("Network Trainer", "opt", "optimizer", "Optimizer: adam, sgd (default: adam)")
      opts.addOption("Network Trainer", "ploss", "policy-loss-weight", "Weight for policy loss (default: 1.0)")
      opts.addOption("Network Trainer", "vloss", "value-loss-weight", "Weight for value loss (default: 1.0)")
   }

   public static func main () throws {
      let opts = OptionParser(help: "Train a neural network model on training data")
      self.registerOptions(opts: opts)
      // parse the command line arguments, now that all options are registered
      opts.parse(tokens: CommandLine.arguments, failOnUnknownOption: true, ignoreHelp: false)

      let gameCount = opts.get(option: "game-count", orElse: 1)
      print("Training network for \(gameCount) games")
   }
}
