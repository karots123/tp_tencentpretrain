import torch.nn as nn
import sys
from tencentpretrain import mpu


class Model(nn.Module):
    """
    Pretraining models consist of three (five) parts:
        - embedding
        - encoder
        - tgt_embedding (optional)
        - decoder (optional)
        - target
    """

    def __init__(self, args, embedding, encoder, tgt_embedding, decoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.tgt_embedding = tgt_embedding
        self.decoder = decoder
        self.target = target

        if "mlm" in args.target and args.tie_weights:
            self.target.mlm.linear_2.weight = self.embedding.word.embedding.weight
        elif "lm" in args.target and args.tie_weights and "word" in self.embedding.embedding_name_list:
            self.target.lm.output_layer.weight = self.embedding.word.embedding.weight
        elif "lm" in args.target and args.tie_weights and "word" in self.tgt_embedding.embedding_name_list:
            self.target.lm.output_layer.weight = self.tgt_embedding.word.embedding.weight
        if self.decoder is not None and args.share_embedding:
            self.tgt_embedding.word.embedding.weight = self.embedding.word.embedding.weight

    def forward(self, src, tgt, seg, tgt_in=None, tgt_seg=None):
        emb = self.embedding(src, seg)
        memory_bank = self.encoder(emb, seg)
        if self.decoder:
            tgt_emb = self.tgt_embedding(tgt_in, tgt_seg)
            memory_bank = self.decoder(memory_bank, tgt_emb, (seg, tgt_seg))

        if tgt_seg is not None:
            loss_info = self.target(memory_bank, tgt, tgt_seg)
        else:
            loss_info = self.target(memory_bank, tgt, seg)

        return loss_info
 #   def load_state_dict(self, state_dict, strict=True):
 #       """Customized load."""

#        if language_model_key in state_dict:
#            state_dict = state_dict[language_model_key]
 #       if self.embedding:
 #           self.embedding.load_state_dict(state_dict, strict=strict)
 #       if self.encoder:
 #           self.encoder.load_state_dict(state_dict, strict=strict)
 #       if self.tgt_embedding:
 #           self.tgt_embedding.load_state_dict(state_dict, strict=strict)
 #       if self.decoder:
 #           self.decoder.load_state_dict(state_dict, strict=strict)
 #       if self.target:
 #           self.target.load_state_dict(state_dict, strict=strict)



class MegatronModule(nn.Module):
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    def __init__(self, share_word_embeddings=True):
        super(MegatronModule, self).__init__()
        self.share_word_embeddings = share_word_embeddings

    def state_dict_for_save_checkpoint(
        self, destination=None, prefix="", keep_vars=False
    ):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(destination, prefix, keep_vars)

    def word_embeddings_weight(self):
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            return self.language_model.embedding.word_embeddings.weight
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            if not self.share_word_embeddings:
                raise Exception(
                    "word_embeddings_weight() called for last "
                    "stage, but share_word_embeddings is false"
                )
            return self.word_embeddings.weight
        raise Exception(
            "word_embeddings_weight() should be " "called for first and last stage only"
        )

    def initialize_word_embeddings(self, init_method_normal):
        args = get_args()
        if not self.share_word_embeddings:
            raise Exception(
                "initialize_word_embeddings() was called but "
                "share_word_embeddings is false"
            )

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism. If we aren't using pipeline
        # parallelism there is nothing to do.
        if args.pipeline_model_parallel_size == 1:
            return

        # Parameters are shared between the word embeddings layer, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        if mpu.is_pipeline_last_stage():
            assert not mpu.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = "word_embeddings_for_head"
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.word_embeddings = mpu.VocabParallelEmbedding(
                args.padded_vocab_size,
                args.hidden_size,
                init_method=init_method_normal(args.init_method_std),
            )
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True

        # Ensure that first and last stages have the same initial parameter
        # values.
        if torch.distributed.is_initialized():
            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                torch.distributed.all_reduce(
                    self.word_embeddings_weight().data, group=mpu.get_embedding_group()
                )
        else:
            print(
                "WARNING! Distributed processes aren't initialized, so "
                "word embeddings in the last layer are not initialized. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )



class MPModel(MegatronModule):
    """
    Pretraining models consist of three (five) parts:
        - embedding
        - encoder
        - tgt_embedding (optional)
        - decoder (optional)
        - target
    """

    def __init__(self, args, embedding, encoder, tgt_embedding, decoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.tgt_embedding = tgt_embedding
        self.decoder = decoder
        self.target = target

        if "mlm" in args.target and args.tie_weights:
            self.target.mlm.linear_2.weight = self.embedding.word.embedding.weight
        elif "lm" in args.target and args.tie_weights and "word" in self.embedding.embedding_name_list:
            self.target.lm.output_layer.weight = self.embedding.word.embedding.weight
        elif "lm" in args.target and args.tie_weights and "word" in self.tgt_embedding.embedding_name_list:
            self.target.lm.output_layer.weight = self.tgt_embedding.word.embedding.weight
            
        if self.decoder is not None and args.share_embedding:
            self.tgt_embedding.word.embedding.weight = self.embedding.word.embedding.weight

    def forward(self, src, tgt, seg, tgt_in=None, tgt_seg=None):
        emb = self.embedding(src, seg)
        memory_bank = self.encoder(emb, seg)
        if self.decoder:
            tgt_emb = self.tgt_embedding(tgt_in, tgt_seg)
            memory_bank = self.decoder(memory_bank, tgt_emb, (seg, tgt_seg))

        if tgt_seg is not None:
            loss_info = self.target(memory_bank, tgt, tgt_seg)
        else:
            loss_info = self.target(memory_bank, tgt, seg)

        return loss_info