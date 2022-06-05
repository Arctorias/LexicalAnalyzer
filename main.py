
import os

from src.inputHandler.TokensInputHandler  import TokensInputHandler
from src.inputHandler.LexemesInputHandler import LexemesInputHandler
from src.inputHandler.UserInputHandler    import UserInputHandler
from src.core.LexicalAnalyzer             import LexicalAnalyzer
from src.outputHandler.OutputHandler      import OutputHandler

_dirname = os.path.dirname(__file__)+"/"

AVAILABLE_LEXEMES_AUTOMATA = _dirname+"syntaticAnalysis/automatons/lexemes.txt"
LEXEMES_AUTOMATA_PATH      = _dirname+"syntaticAnalysis/automatons/json/"
INPUT_PATH                 = _dirname+"expr/input.txt"
TOKEN_PATH                 = _dirname+"lexicalAnalysis/tokens/"
AVAILABLE_TOKENS_AUTOMATA  = _dirname+"lexicalAnalysis/automatons/tokens.txt"
TOKENS_AUTOMATA_PATH       = _dirname+"lexicalAnalysis/automatons/json/"

OUTPUT_PATH                = _dirname+"output/"

if __name__ == "__main__":
    
    # lexemes_input_handler = LexemesInputHandler(LEXEMES_AUTOMATA_PATH, AVAILABLE_LEXEMES_AUTOMATA)
    # lexemes               = lexemes_input_handler.get_lexemes()

    print("reading input")
    user_input_handler    = UserInputHandler(INPUT_PATH)
    input                 = user_input_handler.get_input()

    tokens_input_handler  = TokensInputHandler(TOKEN_PATH,AVAILABLE_TOKENS_AUTOMATA, TOKENS_AUTOMATA_PATH)
    tokens                = tokens_input_handler.get_tokens()
    tokens_automatons     = tokens_input_handler.get_tokens_automatons()

    print("running lexical analyzer")
    analyzer              = LexicalAnalyzer(input, tokens, tokens_automatons)
    automaton_and_token_table = analyzer.tokenize()

    generated_automaton   = automaton_and_token_table[0]
    generated_token_table = automaton_and_token_table[1]

    print("generating output")
    output_handler        = OutputHandler(OUTPUT_PATH, generated_automaton, generated_token_table)
    output_handler.generate_automaton_output()
    output_handler.generate_token_table_output()
