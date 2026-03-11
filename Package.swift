// swift-tools-version: 5.12

import PackageDescription

let package = Package(
    name: "Paw",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(name: "Paw", targets: ["Paw"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/apple/swift-argument-parser.git",
            from: "1.3.0"
        ),
        .package(
            url: "https://github.com/vapor/vapor.git",
            from: "4.0.0"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            branch: "main"
        ),
    ],
    targets: [
        .executableTarget(
            name: "Paw",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Vapor", package: "vapor"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
            ]
        ),
    ]
)
