import copy

DEL    = "del"
OP     = "op"
LIT    = "lit"
ID     = "id"
CMM    = "cmm"
NUM    = "num"
STRING = "string"
TRUE   = "true"
FALSE  = "false"
INT    = "int"
FLOAT  = "float"
class LexicalAnalyzer:
    def __init__(self, input, tokens, tokens_automatons):
        self._input             = input
        self._tokens            = tokens
        self._ordered_tokens    = list()
        self._token_table       = {}
        self._tokens_automatons = tokens_automatons
        self._found_tokens      =  {"kw":[], "id":[], "del":[], "op":[], "cmm":[], "lit":[], "num":[]}

        self._generated_automaton = list()


    def tokenize(self):
        # try:
        self.find_tokens()
        self.generate_automaton()
        # except KeyError as e:

        #     raise RuntimeError("An error ocurred whilst analizing")
        return [self._generated_automaton, self._token_table]
    
    def generate_automaton(self):

        token_index   = 0
        token_name    = self._ordered_tokens[token_index]
        current_token = self.retrieve_token(token_name)
        current_state = current_token.initial_state()

        formatted_input = self._input

        formatted_input = formatted_input.replace(" ", "")

        for index in range(len(formatted_input)):
            char          = formatted_input[index]
            if char == '"':
                char = "'"
            self._generated_automaton.append(current_state)

            if current_state.is_final() and not current_state.has_production_with_char(char):
                token_index += 1
                next_token_structure = self._ordered_tokens[token_index]
                next_token      = self.retrieve_token(next_token_structure)
                next_state      = next_token.initial_state()

                # the two calls below are what is making the automata union
                current_state.update_final_state(char, next_state.name())
                next_state.update_initial_state()
                current_token = next_token
                current_state = next_state
            
            next_state_name = current_state.next_state_by_char(char)
            current_state   = current_token.get_state(next_state_name)
        
        self._generated_automaton.append(current_state)
            

    def retrieve_token(self, next_token_structure):
        if next_token_structure[0] == ID or next_token_structure[0] == CMM:
            token_name = next_token_structure[0]
        elif next_token_structure[0] == LIT:
            if next_token_structure[1][0] == '"' and next_token_structure[1][len(next_token_structure[1])-1] == '"':
                token_name = STRING
            elif next_token_structure[1] == TRUE:
                token_name = TRUE
            elif next_token_structure[1] == FALSE:
                token_name = FALSE
            elif next_token_structure[1] == INT:
                token_name = INT
            elif next_token_structure[1] == FLOAT:
                token_name = FLOAT
            else:
                token_name = LIT
        elif next_token_structure[0] == NUM:
            try:
                if "." in next_token_structure[1]:
                    float(next_token_structure[1])
                else:
                    int(next_token_structure[1])
                token_name = NUM
            except:
                if next_token_structure[0][0] == '"':
                    token_name = STRING
                else:
                    token_name = next_token_structure[1]
        else:
            token_name = next_token_structure[1]
        token = self._tokens_automatons[token_name]
        return copy.deepcopy(token)

    def find_tokens(self):

        self._found_agraggate = False
        word = ""
        for index in range(len(self._input)):
            if self._found_agraggate:
                # shit workaround to deal with 2 char operators, such as ==, <=, etc.
                self._found_agraggate = False
                continue

            char = self._input[index]
            if index < len(self._input)-1:
                next_char = self._input[index+1]

            if char in self._tokens[DEL] or char in self._tokens[OP] or char == " ":
                token = self.find_token(word)
                if token is not None:
                    self._ordered_tokens.append([token[0], word])
                    self._token_table[word] = token
                token = self.find_token(char, next_char)
                if token is not None:
                    self._ordered_tokens.append([token[0],token[1]])
                    self._token_table[token[1]] = token
                word = ""
            else:
                word +=char
            
    def find_token(self, word, next_char=None):
        if word == " " or word =="":
            return
        
        #dealing with strings
        if word[0] == '"' and word[len(word)-1] == '"':
            return [LIT, word]

        # dealing with operators, delimiters and keywords
        for token in self._tokens:
            array = self._tokens[token]
            if word in array:

                # the four rules below are for cases with op with 2 chars, such as ==, <=, etc.
                if token == DEL and next_char == "=":
                    word += "="
                    token = OP
                elif token == OP and next_char == "-":
                    self._found_agraggate = True
                    word += "-"
                elif token == OP and next_char == "+":
                    self._found_agraggate = True
                    word += "+"
                elif token == OP and (word == "<" or word == ">" or word == "!") and next_char == "=":
                    self._found_agraggate = True
                    word += "="
                self._found_tokens[token].append(word)
                return [token, word]
        
        try:
            #dealing with num
            int(word[0])
            return [NUM, word]
        except:
            #dealing with identifiers
            self._found_tokens[ID].append(word)
            return [ID, word]