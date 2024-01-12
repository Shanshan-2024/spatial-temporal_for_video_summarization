import numpy as np

tensor_shape = (15, 1024)
tensor = np.random.rand(*tensor_shape)  # 生成一个随机张量，这里仅作示例

group_size = 12
overlap = 2
total_groups = (tensor_shape[0] - overlap) // (group_size - overlap)

last_group_elements = tensor_shape[0] - total_groups * (group_size - overlap)
if last_group_elements < group_size:
    padding_elements = group_size - last_group_elements
    padding = np.zeros((padding_elements, tensor_shape[1]))
    tensor = np.vstack([tensor, padding])
    total_groups = total_groups + 1

result = []
for i in range(total_groups):
    start = i * (group_size - overlap)
    end = start + group_size
    result.append(tensor[start:end])
    print(tensor[start:end].shape)

result = np.array(result)  #[2, 12, 1024]
print(result.shape)