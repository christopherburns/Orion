import Swift
import Darwin
import Foundation

public class Option {

   var category : String = ""
   var shortName : String = ""
   var longName : String = ""
   var doc : String = ""
   var requireArgument : Bool = true

   init (_ category : String, _ short : String, _ long : String, _ doc : String, requireArgument : Bool = true) {
      self.category = category
      self.shortName = short
      self.longName = long
      self.doc = doc
      self.requireArgument = requireArgument
   }
}

public class ToggleOption : Option {
   init (_ category : String, _ long: String, _ doc: String) {
      super.init(category, "", long, doc, requireArgument: false)
   }
}

/// Returns true if the token parses as a negative numeric literal (e.g. "-1", "-42", "-1.0", "-0.5"),
/// in which case it should be treated as an argument value, not an option flag.
private func isNegativeNumberToken (_ token: String) -> Bool {
   guard token.first == "-", token.count > 1 else { return false }
   // Rely on Swift's standard library numeric parsing so we don't have to
   // re-specify the grammar for numeric literals here.
   if Double(token) != nil { return true }
   if Int(token) != nil { return true }
   return false
}

func == (lhs : Option, rhs : Option) -> Bool {
   return lhs.category == rhs.category && lhs.shortName == rhs.shortName && lhs.longName == rhs.longName
}

public class OptionParser {

   public var commandLineArgs : [String] = [] // original command line provided by client

   public var helpMessage : String = ""

   /*
      Depending on the format the user input is provided, it is classified as follows:
      - option    EX. -<option_short_name> <option value> OR --<option_long_name> <option value>
         - parsed is a dict stored the key value pairs
      - machine parameters EX. --machine-override <key>:<val>
      - argument  EX. <argument>
   */
   public var options   : [ Option ] = []
   public var toggleOptions : [ ToggleOption ] = []
   public var parsedOptions : [ String : [String] ] = [:]
   public var parsedToggleOptions : [ String : Bool ] = [:]
   public var arguments : [ String ] = []


   //////////////////////////////
   // Properties and Accessors //
   //////////////////////////////

   var numArguments : Int { return arguments.count }

   public func getArgument (index : Int) -> String {
      return arguments[index]
   }

   public func wasProvided (option : String) -> Bool {
      for o in parsedOptions.keys {
         if option == o {
            return true
         }
      }
      for o in parsedToggleOptions.keys {
         if option == o {
            return true
         }
      }
      for o in arguments {
         if option == o {
            return true
         }
      }
      return false
   }


   //////////////////////
   // Option Accessors //
   //////////////////////

   // Get option as an optional value - this is the most general accessor, returns nil
   // if the option is not provided or the value is not convertible to the type T
   public func get<T: LosslessStringConvertible> (option: String, atArgument: Int = 0) -> T? {
      if let args = parsedOptions[option] {
         if let g = T(args[atArgument]) {
            return g
         }
      }
      return nil
   }

   public func get (option : String, orElse : UInt64, atArgument : Int = 0) -> UInt64 {
      if let args = parsedOptions[option] {
         if let g = UInt64(args[atArgument]) {
            return g
         }
      }
      return orElse
   }

   public func get (option : String, orElse : Int, atArgument : Int = 0) -> Int {
      if let args = parsedOptions[option] {
         if let g = Int(args[atArgument]) {
            return g
         }
      }
      return orElse
   }

   public func get (option : String, orElse : Float, atArgument : Int = 0) -> Float {
      if let args = parsedOptions[option] {
         if let g = Float(args[atArgument]) {
            return g
         }
      }
      return orElse
   }

   public func get (option : String, orElse : Bool, atArgument : Int = 0) -> Bool {
      if let args = parsedOptions[option] {
         if let g = Bool(args[atArgument]) {
            return g
         }
      }
      return orElse
   }

   public func get (option : String, orElse : String, atArgument : Int = 0) -> String {
      if let args = parsedOptions[option] {
         return String(args[atArgument])
      }
      else {
         return orElse
      }
   }

   public func getToggleOption (_ option : String, orElse : Bool) -> Bool {
      if let value = parsedToggleOptions[option] {
         return value
      }
      else {
         return orElse
      }
   }


   public func getAll<T>(option: String, as type: T.Type, orElse: [T] = []) -> [T] {
      if let args = parsedOptions[option] {
         return args.compactMap {
            if let value = $0 as? T {
               return value
            } else if T.self == Int.self, let intValue = Int($0) as? T {
               return intValue
            } else if T.self == Float.self, let floatValue = Float($0) as? T {
               return floatValue
            } else if T.self == String.self {
               return String($0).trimmingCharacters(in: .whitespacesAndNewlines) as? T
            } else if T.self == Bool.self, let boolValue = Bool($0) as? T {
               return boolValue
            }
            return nil
         }
      } else {
         return orElse
      }
   }

   public init (help : String = "[No Help Message]") {
      self.helpMessage = help
      addOption("General", "h", "help", "Displays this help message")
   }

   public var description : String {
      var s = ""
      s += "options   = \(options)\n"
      s += "arguments = \(arguments)\n"
      s += "parsed    = \(parsedOptions)\n"
      return s
   }

   public func addOption (_ category : String, _ shortFlag : String, _ longFlag : String, _ doc : String, requireArgument: Bool = true) {
      options.append(Option(category, shortFlag, longFlag, doc, requireArgument: requireArgument))
   }

   public func addToggleOption (_ category : String, _ long : String, _ doc : String) {
      toggleOptions.append(ToggleOption(category, long, doc))
   }


   // parse will throw out any previously parsed data and consume a token list from a
   // new command line.
   @discardableResult public func parse (tokens : [String], failOnUnknownOption: Bool, ignoreHelp : Bool = false) -> OptionParser {

      // tokens is an array of strings delimited by whitespace
      self.commandLineArgs = tokens
      self.arguments = []
      self.parsedToggleOptions = [:]
      self.parsedOptions = [:]

      // Walk through tokens, recording arguments until the first option
      argumentParsing : for token in tokens {
         if !token.hasPrefix("-") {
            self.arguments.append(token)
         } else {
            break argumentParsing
         }
      }


      // Parse toggle options
      for t in 0 ..< tokens.count {
         // look for --enable-<key> or --disable-<key>
         if tokens[t].contains("--enable") || tokens[t].contains("--disable") {
            for option in toggleOptions {
               if tokens[t] == ("--enable-" + option.longName) {
                  print ("Enabling \(option.longName)")
                  parsedToggleOptions[option.longName] = true
               }
               else if tokens[t] == ("--disable-" + option.longName) {
                  print ("Disabling \(option.longName)")
                  parsedToggleOptions[option.longName] = false
               }
            }
         }
      }

      // Now look for options
      for t in 0 ..< tokens.count {

         let validOptsShort = options.map({ $0.shortName });
         let validOptsLong  = options.map({ $0.longName });

         let token = tokens[t]
         let isToggleOption = token.contains("--enable") || token.contains("--disable")

         // Treat tokens that look like negative numbers (e.g. "-1") as arguments, not options.
         let isPotentialOption = token.first == "-" && !isToggleOption && !isNegativeNumberToken(token)

         if failOnUnknownOption && isPotentialOption {
            let startIndex = Array(token)[1] == "-" ? 2 : 1
            let optionName = String(token.dropFirst(startIndex))
            // Ensure option provided is valid and is nonempty

            if (optionName.count == 0 || (!validOptsShort.contains(optionName) && !validOptsLong.contains(optionName))) {
               print ("Error: Unrecognized option provided: \"\(token)\". Refusing to continue...");
               usage()
               exit(0);
            }
         }

         // We know the token is valid. Just figure out which one and collect arguments
         for option in options {
            if token == ("--" + option.longName) || token == ("-" + option.shortName) {
               // We have a match. collect args until we hit the next option
               var args : [String] = []
               collectArgs : for a in (t+1) ..< tokens.count {
                  let nextToken = tokens[a]
                  // Treat tokens that look like negative numbers as arguments, even though they start with '-'
                  if !nextToken.hasPrefix("-") || isNegativeNumberToken(nextToken) {
                     args.append(tokens[a])
                  } else {
                     break collectArgs
                  }
               }

               parsedOptions[option.longName] = args

               // Validate that options requiring values actually received them
               if args.isEmpty && option.requireArgument {
                  print("Error: Option '\(option.longName)' requires a value but none was provided")
                  usage()
                  exit(1)
               }
            }
         }
      }

      /// Finally, check to see if the help option was specified, and if so, present
      /// the help message and bail
      if !ignoreHelp && wasProvided(option: "help") {
         usage()
         exit(0)
      }

      return self
   }


   public func usage () {

      var longNameFieldWidth = 0
      var shortNameFieldWidth = 0
      for o in options {
         shortNameFieldWidth = max(shortNameFieldWidth, o.shortName.count)
         longNameFieldWidth = max(longNameFieldWidth, o.longName.count)
      }
      for to in toggleOptions {
         longNameFieldWidth = max(longNameFieldWidth, ("{enable,disable}-" + to.longName).count)
      }

      shortNameFieldWidth += 1  // one dash
      longNameFieldWidth += 2   // two dashes

      print ("\(helpMessage)")

      let allCategories = Set(options.map { $0.category } + toggleOptions.map { $0.category })
      var uniqueCategories = Array(allCategories)

      // Crudely move "General" to the front of the list
      if uniqueCategories.contains("General") {
         uniqueCategories.remove(at: uniqueCategories.firstIndex(of: "General")!)
         uniqueCategories.insert("General", at: 0)
      }

      for category in uniqueCategories {
         print ("\n\(category)\n")
         for o in options {
            if o.category == category {
               let shortFieldString = ((o.shortName != "" ? "-" : "") + o.shortName).leftPadded(to: shortNameFieldWidth)
               let longFieldString  = ("--" + o.longName).leftPadded(to: longNameFieldWidth)
               print ("  \(shortFieldString)  \(longFieldString)    \(o.doc)")
            }
         }

         for to in toggleOptions {
            if to.category == category {
               let shortFieldString = String(repeating: " ", count: shortNameFieldWidth)
               let longFieldString  = ("--{enable,disable}-" + to.longName).leftPadded(to: longNameFieldWidth)
               print ("  \(shortFieldString)  \(longFieldString)    \(to.doc)")
            }
         }
      }
   }

   public func copy () -> OptionParser {
      let newParser = OptionParser(help: self.helpMessage)
      newParser.commandLineArgs = self.commandLineArgs
      newParser.options = self.options.map { Option($0.category, $0.shortName, $0.longName, $0.doc) }
      newParser.toggleOptions = self.toggleOptions.map { ToggleOption($0.category, $0.longName, $0.doc) }
      newParser.parsedOptions = self.parsedOptions
      newParser.parsedToggleOptions = self.parsedToggleOptions
      newParser.arguments = self.arguments
      return newParser
   }
}

extension String {
   func leftPadded (to length: Int) -> String {
      if self.count >= length {
         return self
      }
      return String(repeating: " ", count: length - self.count) + self
   }
}

