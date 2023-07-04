import numpy
import scipy.special
class neuralNetwork:
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learingrate):
        # 输入层, 隐藏层, 输出层的数量, 学习率
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 权重初始化, 根据正太分布来做 
        # 我们将正态分布的中心设定为0.0。与下一层中节点相关的标准方差的表达式，
        # 按照Python的形式，就是pow(self.hnodes, -0.5)，
        # 简单说来，这个表达式就是表示节点数目的-0.5次方。最后一个参数，就是我们希望的 numpy数组的形状大小。
        # 权重矩阵 wi,j  i代表当前的节点, j 代表下一个节点. 例如下一排有 3 个节点, i.1 i.2 i.3 所以会有 3 行
        # 当前行有 4 个节点, 所以会存在 wi 1,  wi 2,  wi 3,  wi 4, 每一行代表一个当前节点
        self.wih = numpy.random.normal(0.0, pow(self.hnodes , -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)) 

        self.lr = learingrate

        # 激活函数为 s 函数
        self.activation_function = lambda x: scipy.special.expit(x)

        # print(f"input to hidden 权重矩阵:\n {self.wih}")
        # print(f"hidden to output 权重矩阵:\n {self.who}")
    
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        # print(f"hidden_inputs:\n{hidden_inputs}")
        # print(f"wih: {self.wih} * inputs: {inputs} = hidden_inputs: {hidden_inputs}")
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_erros = targets - final_outputs
        # 误差需要和权重相乘才是 隐藏层的误差
        hidden_erros = numpy.dot(self.who.T, output_erros)

        # 更新权重
        self.who += self.lr * numpy.dot((output_erros * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_erros * hidden_outputs * (1 - hidden_outputs)),  numpy.transpose(inputs))
        
        

    # query 函数接受神经网络的输入，返回网络的输出
    def query(self, inputs_list):
        
        inputs = numpy.array(inputs_list, ndmin=2).T
        # print(f"inputs:\n{inputs}")

        hidden_inputs = numpy.dot(self.wih, inputs)
        # print(f"hidden_inputs:\n{hidden_inputs}")
        # print(f"wih: {self.wih} * inputs: {inputs} = hidden_inputs: {hidden_inputs}")
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


if __name__ == "__main__":
    input_nodes = 28 * 28
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1
    
    # targets = numpy.zeros(onodes) + 0.01 
    # targets[int(all_values[0])] = 0.99
    n = neuralNetwork(inputnodes=input_nodes, hiddennodes=hidden_nodes, outputnodes=output_nodes, learingrate=learning_rate)
    # n.query([0.1, 0.2, 0.3, 0.2])

    trainnig_data_file = open("mnist_dataset/mnist_train.csv", 'r')
    trainning_data_list = trainnig_data_file.readlines()
    trainnig_data_file.close()
    epochs = 7
    for e in range(epochs):
        idx = 0
        print(f"epoch: {e}")
        for record in trainning_data_list:
            idx += 1
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
    
    test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scorecard = []
    idx = 0
    for record in test_data_list:
        print(f"test id {idx}")
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        corrent_label = int(all_values[0])
        print("correct_label", corrent_label)
        # print(f"iiputs: {inputs}, {all_values}")
        outputs = n.query(inputs)

        # 根据我们定义的输出, 最大的那个值就是我们的 label 
        got_label = numpy.argmax(outputs)
        print(f"got_label: {got_label}")
        if got_label == corrent_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

        scorecard_array = numpy.array(scorecard) 
        performance = scorecard_array.sum() / scorecard_array.size
        print(f"performance = {performance}")
    
