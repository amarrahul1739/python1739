from moon import halfmoon_shuffle
from perceptron import perceptron
from cs import get_inputs

def main():
    number_tr = 2000
    number_ts = 1000
    radius = 10
    distance = 0
    width = 7
    number_samp = number_tr + number_ts

    print("initalizing half-moon\n")
    data =  halfmoon_shuffle(radius,width,distance,number_samp)

    print("number of samples generated= %d\n" %number_samp)
    print("radius= %2.1f\n" % radius)
    print("width of the half-moon=%2.1f\n" %width)
    print("distance=%2.1f\n" %distance)

    learn_rate = 0.01
    epochs = 50
    neuron = 2
    bias = distance/2

    perecep = perceptron(data,learn_rate,neuron,bias,epochs)
    perecep.train(number_tr)
    perecep.test(number_ts,number_tr)


if __name__ == "__main__":
    main()