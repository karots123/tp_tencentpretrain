import math
import torch.nn as nn
from tencentpretrain import mpu
import torch
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

class WordEmbedding(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(WordEmbedding, self).__init__()
        self.mp = args.use_mp
        if self.mp:
            self.embedding = mpu.VocabParallelEmbedding(vocab_size, args.emb_size)

        else:
            self.embedding = nn.Embedding(vocab_size, args.emb_size)
        self.emb_size = args.emb_size
        self.sinusoidalpos = False
        if "sinusoidalpos" in args.embedding:
            self.sinusoidalpos = True


    def forward(self, src, _):
        """
        Args:
            src: [batch_size x seq_length]
            seg: [batch_size x seq_length]
        Returns:
            emb: [batch_size x seq_length x hidden_size]
        """

        emb = self.embedding(src)
        if self.sinusoidalpos:
            return emb * math.sqrt(self.emb_size)
        else:
            return emb


#    def load_state_dict(self, state_dict, strict=True):
#        """Customized load."""

        # Word embedding.
#        if self._word_embeddings_key in state_dict:
#            state_dict_ = state_dict[self._word_embeddings_key]
#        else:
            # for backward compatibility.
#            state_dict_ = {}
#            for key in state_dict.keys():
#                if 'word_embeddings' in key:
#                    state_dict_[key.split('word_embeddings.')[1]] \
#                        = state_dict[key]
#        print(get_tensor_model_parallel_world_size())
#        vocab_len = state_dict_['weight'].shape[0]
#        state_dict_["weight"] = state_dict_["weight"][:vocab_len// get_tensor_model_parallel_world_size()]
#        self.embedding.load_state_dict(state_dict_, strict=strict)

       
