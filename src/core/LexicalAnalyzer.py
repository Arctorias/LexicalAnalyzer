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
        self.find_tokens()
        self.generate_automaton()

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
                next_token_name = self._ordered_tokens[token_index]
                next_token      = self.retrieve_token(next_token_name)
                next_state      = next_token.initial_state()
                current_state.update_final_state(char, next_state.name())
                next_state.update_initial_state()
                current_token = next_token
                current_state = next_state
            
            next_state_name = current_state.next_state_by_char(char)
            current_state   = current_token.get_state(next_state_name)
        
        self._generated_automaton.append(current_state)
            

    def retrieve_token(self, name):
        if name[0] == ID or name[0] == CMM:
            token_name = name[0]
        elif name[0] == LIT:
            if name[1][0] == '"' and name[1][len(name[1])-1] == '"':
                token_name = STRING
            elif name[1][0] == "t":
                token_name = TRUE
            elif name[1][0] == "f":
                token_name = FALSE
            else:
                token_name = LIT
        elif name[0] == NUM:
            try:
                if "." in name[1]:
                    float(name[1])
                else:
                    int(name[1])
                token_name = NUM
            except:
                if name[0][0] == '"':
                    token_name = STRING
                else:
                    token_name = name[1]
        else:
            token_name = name[1]
        token = self._tokens_automatons[token_name]
        return copy.deepcopy(token)

    def find_tokens(self):
        word = ""
        for index in range(len(self._input)):
            char = self._input[index]

            if char in self._tokens[DEL] or char in self._tokens[OP] or char == " ":
                token = self.find_token(word)
                if token is not None:
                    self._ordered_tokens.append([token,word])
                    self._token_table[word] = token
                token = self.find_token(char)
                if token is not None:
                    self._ordered_tokens.append([token,char])
                    self._token_table[char] = token
                word = ""
            else:
                word +=char
            
    def find_token(self, word):
        if word == " " or word =="":
            return
        
        #dealing with strings
        if word[0] == '"' and word[len(word)-1] == '"':
            return LIT

        # dealing with operators, delimites and keywords
        for token in self._tokens:
            array = self._tokens[token]
            if word in array:
                self._found_tokens[token].append(word)
                return token
        
        #dealing with identifiers

        try:
            int(word[0])
            return NUM
        except:
            self._found_tokens[ID].append(word)
            return ID