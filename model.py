import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

# from peft import (
#     LoraConfig,
#     get_peft_model,
#     prepare_model_for_int8_training,
#     set_peft_model_state_dict
# )
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training, 
    set_peft_model_state_dict,
)


class LLM4Rec(nn.Module):
    def __init__(self, **args):
        super(LLM4Rec, self).__init__()
        self.args = args
        self.input_dim, self.output_dim = args['input_dim'], args['output_dim']

        print(f'Initializing language decoder ...')
        # add the lora module
        peft_config = LoraConfig(
            task_type='FEATURE_EXTRACTION',
            r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            target_modules=self.args['lora_target_modules'],
            bias='none',
        )

        # model_path = "/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/hf_cache/hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549"
        model_path = self.args['base_model']
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # )
        # self.llama_model = AutoModel.from_pretrained(
        #     model_path,
        #     quantization_config=bnb_config,
        #     dtype=torch.float16,
        #     cache_dir=args['cache_dir'],
        #     device_map=self.args['device_map'],
        # )

        _dtype = torch.float32 if self.args.get('device_map') == 'cpu' else torch.float16
        self.llama_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=_dtype,
            cache_dir=args['cache_dir'] or None,
            device_map=self.args['device_map'],
        )
        self.llama_model.enable_input_require_grads()  # needed for LoRA + grad checkpointing
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # self.llama_model = prepare_model_for_int8_training(self.llama_model)
        # self.llama_model = prepare_model_for_kbit_training(self.llama_model)
        # self.llama_model = prepare_model_for_kbit_training(
        #                         self.llama_model,
        #                         use_gradient_checkpointing=False
        #                     )
        # self.llama_model = get_peft_model(self.llama_model, peft_config)
        # self.llama_model.enable_input_require_grads()
        # self.llama_model.gradient_checkpointing_enable(
        #     gradient_checkpointing_kwargs={"use_reentrant": False}
        # )
        self.llama_model.print_trainable_parameters()
        self.llama_model.config.use_cache = False

        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            cache_dir=args['cache_dir'],
        )
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        self.instruct_ids, self.instruct_mask = self.llama_tokenizer(self.args['instruction_text'][0],
                                                                     truncation=True, padding=False,
                                                                     return_tensors='pt', add_special_tokens=False).values()
        self.response_ids, self.response_mask = self.llama_tokenizer(self.args['instruction_text'][1],
                                                                     truncation=True, padding=False,
                                                                     return_tensors='pt', add_special_tokens=False).values()
        print('Language decoder initialized.')

        
        self.task_type = args['task_type']
        # if self.task_type == 'general':
        #     self.user_embeds = nn.Embedding.from_pretrained(self.args['user_embeds'], freeze=True)
        #     self.user_proj = nn.Linear(self.input_dim, self.llama_model.config.hidden_size)
        # self.input_embeds = nn.Embedding.from_pretrained(self.args['input_embeds'], freeze=True)
        # self.input_proj = nn.Linear(self.input_dim, self.llama_model.config.hidden_size)

        # get device and dtype from llama_model
        device = next(self.llama_model.parameters()).device
        dtype = next(self.llama_model.parameters()).dtype
        if self.task_type == 'general': 
            self.user_embeds = nn.Embedding.from_pretrained(self.args['user_embeds']).to(device=device, dtype=dtype)
            self.user_proj = nn.Linear(self.input_dim, self.llama_model.config.hidden_size).to(device=device, dtype=dtype)
        self.input_embeds = nn.Embedding.from_pretrained(self.args['input_embeds']).to(device=device, dtype=dtype)
        self.input_proj = nn.Linear(self.input_dim, self.llama_model.config.hidden_size).to(device=device, dtype=dtype)

        self.score = nn.Linear(self.llama_model.config.hidden_size, self.output_dim, bias=False).to(device=device, dtype=dtype)


    def predict(self, inputs, inputs_mask, history_metadata=None):
        bs = inputs.shape[0]
        device = next(self.llama_model.parameters()).device
        embed_tokens = self.llama_model.get_input_embeddings()
        response_embeds = embed_tokens(self.response_ids.to(device)).expand(bs, -1, -1)
        response_mask = self.response_mask.to(device).expand(bs, -1)

        if history_metadata is not None:
            tokens = self.llama_tokenizer(
                history_metadata,
                truncation=True,
                padding=True,
                return_tensors='pt',
                add_special_tokens=False
            ).to(device)
            instruct_embeds = embed_tokens(tokens.input_ids)
            instruct_mask = tokens.attention_mask
        else:
            instruct_embeds = embed_tokens(self.instruct_ids.to(device)).expand(bs, -1, -1)
            instruct_mask = self.instruct_mask.to(device).expand(bs, -1)
        

        if self.task_type == 'general':
            users = self.user_proj(self.user_embeds(inputs[:, 0].unsqueeze(1)))
            items = self.input_proj(self.input_embeds(inputs[:, 1:]))
            inputs = torch.cat([users, items], dim=1)
        else:
            inputs = self.input_proj(self.input_embeds(inputs))
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        outputs = self.llama_model(inputs_embeds=inputs.to(next(self.llama_model.parameters()).dtype), attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.last_hidden_state[:, -1]
        pooled_logits = self.score(pooled_output)
        # pooled_logits = self.score.to(device=pooled_output.device, dtype=pooled_output.dtype)(pooled_output)

        return outputs, pooled_logits.view(-1, self.output_dim)

    def forward(self, inputs, inputs_mask, labels, history_metadata=None):
        outputs, pooled_logits = self.predict(inputs, inputs_mask, history_metadata=history_metadata)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(pooled_logits, labels.view(-1))

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llama_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {"use_reentrant": False}
        )


    def gradient_checkpointing_disable(self):
        self.llama_model.gradient_checkpointing_disable()









