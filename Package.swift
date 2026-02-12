// swift-tools-version: 5.9
import PackageDescription

let package = Package(
   name: "Orion",
   platforms: [
      .macOS(.v14)
   ],
   products: [
      .executable(
         name: "orion",
         targets: ["orion"]
      )
   ],
   dependencies: [
      .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.10.0")
   ],
   targets: [
      .target(
         name: "Core",
         dependencies: ["Utility"],
         path: "src/core"
      ),
      .target(
         name: "Splendor",
         dependencies: [
            "Core", "Utility",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift")
         ],
         path: "src/splendor"
      ),
      .target(
         name: "Utility",
         dependencies: [],
         path: "src/utility"
      ),
      .executableTarget(
         name: "orion",
         dependencies: [
            "Core", "Splendor", "Utility",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXOptimizers", package: "mlx-swift")],
         path: "src",
         sources: ["Orion.swift", "GameplayTester.swift", "DataGenerator.swift", "NetworkTrainer.swift", "SelfPlayTrainer.swift", "Common.swift"]
      ),
   ]
)
