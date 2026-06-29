def verify_special_tokens(tokenizer_path, model_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    
    print("=== Tokenizer Special Tokens ===")
    print(f"pad_token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    print(f"eos_token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    print(f"bos_token: '{tokenizer.bos_token}' (id={tokenizer.bos_token_id})")
    
    print("\n=== Model Config Special Tokens ===")
    print(f"pad_token_id: {model.config.pad_token_id}")
    print(f"eos_token_id: {model.config.eos_token_id}")
    print(f"bos_token_id: {model.config.bos_token_id}")
    
    # 验证一致性
    assert tokenizer.pad_token_id == model.config.pad_token_id, "PAD mismatch!"
    assert tokenizer.eos_token_id == model.config.eos_token_id, "EOS mismatch!"
    
    # 验证padding功能
    print("\n=== Padding Test ===")
    texts = ["测试文本", "短"]
    encoded = tokenizer(texts, padding=True, return_tensors='pt')
    print(f"Input IDs shape: {encoded['input_ids'].shape}")
    print(f"Attention mask:\n{encoded['attention_mask']}")
    
    # 验证生成时不会报错
    print("\n=== Generation Test ===")
    input_ids = encoded['input_ids'][:, :5]  # 取前5个token
    output = model.generate(input_ids, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id)
    print(f"Generated: {tokenizer.batch_decode(output, skip_special_tokens=True)}")
    
    print("\n✅ All checks passed!")

# 执行验证
verify_special_tokens(
    './Qwen3-8B-Base-Char',  # tokenizer路径
    './Qwen3-8B-Base-Char'   # model路径
)