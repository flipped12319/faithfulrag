from datasets import load_dataset
import re
def chunk_text(text, chunk_size=300):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    chunk_inf = []
    chunk_num = 0
    current = ""

    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += (" " if current else "") + sent
        else:
            chunks.append(current)
            current = sent
            chunk_num = chunk_num + 1

    if current:
        chunks.append(current)
        chunk_num = chunk_num + 1

    chunk_inf.append(chunk_num)
    chunk_inf.append(chunks)

    return chunk_inf
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
dataArr=[]
for i in range(0,1):
    dataArr.append(ds["train"][i])



for i, data in enumerate(dataArr):
    text = ""
    # print(i)
    for sentences in dataArr[i]["context"]["sentences"]:
        for sentence in sentences:
            text += sentence + " "   # 正确写法
    text = text.strip()  # 去掉末尾空格

    dataArr[i]["context"] = text

print(dataArr[0]["context"])
# print(ds["train"][0]["context"]["sentences"])
chunk_inf=chunk_text(dataArr[0]["context"])
print(chunk_inf)
for value in chunk_inf[1]:
    print(value)





