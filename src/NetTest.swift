import Foundation
import MLX
import Splendor

/// Hidden `orion nettest` subcommand: structural verification of the
/// PolicyValueNetwork forward pass, independent of any game or training run.
///
/// The tests here guard the properties that a loss curve cannot: that the
/// card-slot → purchase-logit wiring is exactly 1:1 (a mis-scatter trains
/// fine and plays nonsense), and that save/load/clone reproduce the network
/// bit-for-bit. Run after any change to the network topology.
struct NetTest {

   static func main () throws {
      var failures = 0
      func check (_ name: String, _ ok: Bool, _ detail: String = "") {
         print("  [\(ok ? "PASS" : "FAIL")] \(name)\(detail.isEmpty ? "" : " — \(detail)")")
         if !ok { failures += 1 }
      }

      let inputDim = PolicyValueNetwork.INPUT_DIMENSIONS
      let policyDim = PolicyValueNetwork.POLICY_DIMENSIONS
      // Mirror of the network's encoding-layout constants (they are internal
      // to the Splendor module; recompute from the same public primitives).
      let cardFeatures = Card.ENCODED_SIZE                                   // 12
      let cardBlockStart = 4 * PlayerState.ENCODED_SIZE + 5 + 1 + 15 + 130  // 355
      let visibleCards = 12

      print("nettest: PolicyValueNetwork v\(PolicyValueNetwork.ARCHITECTURE_VERSION), input \(inputDim), policy \(policyDim)")

      // fp32 network, dropout off, deterministic seed
      let network = PolicyValueNetwork(seed: 42, dropoutRate: 0.1, precision: .float32)
      network.train(false)

      func run (_ input: [Float]) -> (policy: [Float], value: Float) {
         let x = MLXArray(input).reshaped([1, inputDim])
         let (p, v) = network.execute(x)
         eval(p, v)
         return (p.asArray(Float.self), v.asArray(Float.self)[0])
      }

      // ── 1. Shapes ────────────────────────────────────────────────────────
      let batchX = MLXArray(Array(repeating: Float(0.3), count: 3 * inputDim)).reshaped([3, inputDim])
      let (bp, bv) = network.execute(batchX)
      eval(bp, bv)
      check("policy shape [3, \(policyDim)]", bp.shape == [3, policyDim], "got \(bp.shape)")
      check("value shape [3, 1]", bv.shape == [3, 1], "got \(bv.shape)")
      let v0 = bv.asArray(Float.self)[0]
      check("value in [-1, 1]", v0 >= -1.0 && v0 <= 1.0, "got \(v0)")

      // ── 2. Slot isolation (the scatter test) ────────────────────────────
      // Perturbing card slot k must change purchase logit k and NO other
      // purchase logit: each purchase logit depends only on its own card's
      // embedding. Trunk logits (12..<48) and value may all change (the
      // perturbation moves the pooled board summary) — that is expected.
      let baseline = run([Float](repeating: 0.25, count: inputDim))
      var isolationOK = true
      var diagonalOK = true
      for k in 0..<visibleCards {
         var probe = [Float](repeating: 0.25, count: inputDim)
         for f in 0..<cardFeatures {
            probe[cardBlockStart + k * cardFeatures + f] = 0.9
         }
         let out = run(probe)
         if out.policy[k] == baseline.policy[k] { diagonalOK = false
            print("       slot \(k): own purchase logit did NOT change") }
         for j in 0..<visibleCards where j != k {
            if out.policy[j] != baseline.policy[j] { isolationOK = false
               print("       slot \(k): leaked into purchase logit \(j)") }
         }
      }
      check("perturbing card k moves purchase logit k", diagonalOK)
      check("perturbing card k leaves other purchase logits bit-identical", isolationOK)

      // Trunk logits should react to the board summary (pooled path is live)
      var probeAll = [Float](repeating: 0.25, count: inputDim)
      for i in cardBlockStart..<(cardBlockStart + visibleCards * cardFeatures) { probeAll[i] = 0.9 }
      let outAll = run(probeAll)
      check("board summary reaches trunk (move-12 logit reacts to cards)",
            outAll.policy[12] != baseline.policy[12])
      check("board summary reaches value head", outAll.value != baseline.value)

      // ── 3. Save / load round trip ────────────────────────────────────────
      let tmpURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("nettest_model_\(getpid())")
      defer { try? FileManager.default.removeItem(at: tmpURL) }
      try network.save(to: tmpURL, metadata: ModelMetadata(
         version: "nettest", architectureVersion: PolicyValueNetwork.ARCHITECTURE_VERSION))
      let (reloaded, _) = try PolicyValueNetwork.load(from: tmpURL, precision: .float32)
      reloaded.train(false)
      let fixedInput = (0..<inputDim).map { Float($0 % 17) / 17.0 }
      let x = MLXArray(fixedInput).reshaped([1, inputDim])
      let (p1, v1) = network.execute(x); eval(p1, v1)
      let (p2, v2) = reloaded.execute(x); eval(p2, v2)
      check("save/load round trip: identical policy",
            p1.asArray(Float.self) == p2.asArray(Float.self))
      check("save/load round trip: identical value",
            v1.asArray(Float.self) == v2.asArray(Float.self))

      // ── 4. Clone ─────────────────────────────────────────────────────────
      let cloned = network.clone()
      cloned.train(false)
      let (p3, v3) = cloned.execute(x); eval(p3, v3)
      check("clone: identical policy", p1.asArray(Float.self) == p3.asArray(Float.self))
      check("clone: identical value", v1.asArray(Float.self) == v3.asArray(Float.self))

      if failures > 0 {
         print("nettest: \(failures) FAILURE(S)")
         exit(1)
      }
      print("nettest: all checks passed")
   }
}
