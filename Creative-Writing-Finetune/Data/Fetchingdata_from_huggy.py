# The reason am downloading is i hate waiting for streaming the data

from datasets import load_dataset,Dataset
stream = load_dataset('agentlans/literotica-reddit-dirty-writing-prompts',name = 'sample_k5000' ,split = 'train',streaming = True).shuffle(seed = 42,buffer_size = 1000)
local_dataset = Dataset.from_list(list(stream.take(5000)))
local_dataset.save_to_disk('literotica-reddit-dirty-writing-prompts')
print("Saved documents:", len(local_dataset))

