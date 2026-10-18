import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from dataclasses import dataclass
import math
from transformers import PretrainedConfig,PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from transformers import GenerationConfig
class CasualAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed,3*config.n_embed,bias=False)
        self.c_proj = nn.Linear(config.n_embed,config.n_embed,bias=False)
        self.c_attn_dropout = nn.Dropout(config.dropout)
        self.c_resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional,"scaled_dot_product_attention")
        if not self.flash:
            print("WARNING:Using slow attention")
            self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size))
                                 .view(1,1,config.block_size,config.block_size))

    def forward(self,x):
        B,T,C = x.shape
        q,k,v = self.c_attn(x).split(self.n_embed,dim=2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=None
                                                                 ,dropout_p = self.dropout.p if self.training else 0
                                                                 ,is_causal=True)

        else:
            att = (q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.c_attn_dropout(att)
            y = att @ v

        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.fc = nn.Linear(config.n_embed,4*config.n_embed,bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embed,config.n_embed,bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed,bias=False)
        self.ln2 = nn.LayerNorm(config.n_embed,bias=False)
        self.attn = CasualAttention(config)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPT_config:
    n_head:int = 8
    n_layer:int = 16
    n_embed:int = 512
    dropout:float = 0.2
    block_size:int = 378
    vocab_size:int = 50000


class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed is not None
        assert config.vocab_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embed),
            wpe = nn.Embedding(config.block_size,config.n_embed),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed,bias=False),
        ))

        self.lm_head = nn.Linear(config.n_embed,config.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn,p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p,mean=0,std=0.01/math.sqrt(2*config.n_layer))

        print("Number of parameters:%2f"%(self.get_num_params()/1e6))


    def get_num_params(self,non_embedding=True):
        n_param =sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_param -= self.transformer.wpe.weight.numel()
        return n_param

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0,std=0.01)
        if isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0,std=0.01)


    def forward(self,input_ids,targets=None,attention_mask = None, **kwargs):
        device = input_ids.device
        B,T = input_ids.size()
        assert T <= self.config.block_size,f"Cannot Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0,T,dtype=torch.long,device=device)
        input_ids = input_ids.long()
        tok_embed = self.transformer.wte(input_ids)
        pos_embed = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_embed + pos_embed)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1).long(),ignore_index=-100)
            # print("ignore_index =", -100, "| targets min =", targets.min().item())

        else:
            logits = self.lm_head(x[:,[-1],:])
            loss = None
        return logits,loss

    def crop_blocksize(self,block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block,'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self,weight_decay,betas,learning_rate,device_type):
        param_dict = { pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,}parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self,ffwdbwd_per_iter,dt):
        N = self.get_num_parameters()
        cfg = self.config
        L,H,Q,T = cfg.n_layer,cfg.n_head,cfg.n_embed//cfg.n_head,cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * ffwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 27.1e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self,idx,max_new_tokens,temperature = 1.0, top_k = None):
        for _ in range(max_new_tokens):
            idx_cond =  idx if idx.size(1) <= self.config.block_size else idx[:,-self.config.block_size:]
            logits,_ = self.forward(idx_cond)
            logits = logits[:,-1,:]/temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            prob = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


class  TinyGPTConfig(PretrainedConfig):
    model_type = "tiny_gpt"
    auto_map = {
        "AutoConfig": "Model.TinyGPTConfig",
        "AutoModelForCausalLM": "Model.TinyGPTForCausalLM",
    }

    def __init__(self,
                 n_head: int = 8,
                 n_layer: int = 16,
                 n_embed: int = 512,
                 block_size: int = 378,
                 dropout: float = 0.2,
                 vocab_size: int = 50000,
                 pad_token_id: int = 0,
                 eos_token_id: int = 1,
                 bos_token_id: int = 1,
                 **kwargs):
        for key in ("pad_token_id", "eos_token_id", "bos_token_id",
                    "is_decoder", "is_encoder_decoder", "tie_word_embeddings"):
            kwargs.pop(key, None)

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=True,
            is_decoder=True,
            is_encoder_decoder=False,
            **kwargs,
        )
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_embed = n_embed
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_position_embeddings = block_size
        self.hidden_size = n_embed
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head

class TinyGPTForCausalLM(PreTrainedModel):
    config_class = TinyGPTConfig
    base_model_prefix = "gpt"
    main_input_name = "input_ids"
    _tied_weights_keys = {"gpt.lm_head.weight": "gpt.transformer.wte.weight"}
    all_tied_weights_keys = {"gpt.lm_head.weight": "gpt.transformer.wte.weight"}
    def __init__(self, config):
        super().__init__(config)

        model_args = {
            "n_layer":config.n_layer,
            "n_head":config.n_head,
            "n_embed":config.n_embed,
            "block_size":config.block_size,
            "vocab_size":config.vocab_size,
        }
        gptconfig = TinyGPTConfig(**model_args)
        self.gpt = GPT(gptconfig)
        self.generation_config = GenerationConfig.from_model_config(config)

    def forward(self,attention_mask = None, input_ids=None,labels = None,**kwargs):
        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            if input_ids is None:
                raise ValueError("input_ids is None")
            logits, loss = self.gpt(input_ids, labels, attention_mask=attention_mask)
            return CausalLMOutput(loss=loss, logits=logits)





