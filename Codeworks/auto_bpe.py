from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))


from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(vocab_size=500, min_frequency = 500, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]",
                                     " ", "#", "<", ">"],\
                     )
                                      


from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()
# tokenizer.add_tokens(["-", "#", "<", ">"])


files = ["output.txt"]
tokenizer.train(files, trainer)

tokenizer.save("/home/sysadm/samapankar/bpe/tokenizer-slp1-2-sanskrit.json")


tokenizer = Tokenizer.from_file("/home/sysadm/samapankar/bpe/tokenizer-slp1-2-sanskrit.json")

text = "<" + "devAH api santuzwAH, SabdAnAm cApratipattiH prApnoti" + ">"

output = tokenizer.encode(text)

print(output.ids)
print(output.tokens)
print(len(output.tokens))
# print(tokenizer.get_vocab())
