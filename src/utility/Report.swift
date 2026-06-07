import Foundation

/// Shared envelope and helpers for structured `--report-json` output.
public enum Report {

   public static let SCHEMA_VERSION = 1

   /// ISO 8601 timestamp with second precision.
   public static func timestamp (_ date: Date = Date()) -> String {
      let formatter = ISO8601DateFormatter()
      formatter.formatOptions = [.withInternetDateTime]
      return formatter.string(from: date)
   }

   /// Encode `value` as pretty-printed JSON with sorted keys and write to `path` atomically.
   /// Creates intermediate directories as needed.
   public static func write<T: Encodable> (_ value: T, to path: String) throws {
      let url = URL(fileURLWithPath: path)
      let dir = url.deletingLastPathComponent()
      if !dir.path.isEmpty {
         try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true, attributes: nil)
      }
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      let data = try encoder.encode(value)
      try data.write(to: url, options: .atomic)
   }

   /// Classify an agent CLI spec into a kind tag for structured reports.
   public static func agentKind (spec: String) -> String {
      switch spec.lowercased() {
      case "random":    return "random"
      case "heuristic": return "heuristic"
      case "human":     return "human"
      default:          return "neural"
      }
   }

   /// Short display label for an agent: builtin name lowercased, or model directory basename.
   public static func agentLabel (spec: String) -> String {
      let lowered = spec.lowercased()
      if lowered == "random" || lowered == "heuristic" || lowered == "human" {
         return lowered
      }
      let trimmed = spec.hasSuffix("/") ? String(spec.dropLast()) : spec
      let url = URL(fileURLWithPath: trimmed)
      let last = url.lastPathComponent
      return last.isEmpty ? trimmed : last
   }
}
