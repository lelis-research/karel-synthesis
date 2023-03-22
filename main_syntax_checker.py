import torch
from dsl import DSL
from dsl.syntax_checker import PySyntaxChecker

if __name__ == '__main__':

    dsl = DSL.init_default_karel()
    sample_program = [0, 1, 2, 49, 32, 41, 33, 47, 31, 13, 9, 8, 10, 4, 4, 48, 3]

    syntax_checker = PySyntaxChecker(dsl.t2i, torch.device('cpu'))
    initial_state = syntax_checker.get_initial_checker_state()
    sequence_mask = syntax_checker.get_sequence_mask(initial_state, sample_program).squeeze()
    for idx, token in enumerate(sample_program):
        valid_tokens = torch.where(sequence_mask[idx] == 0)[0]
        valid_tokens = [dsl.i2t[tkn.detach().cpu().numpy().tolist()] for tkn in valid_tokens]
        valid_tokens = " ".join(valid_tokens)
        print("valid tokens for {}: {}".format(dsl.i2t[token], valid_tokens))