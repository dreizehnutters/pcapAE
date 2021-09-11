import copy
from . import utils
class LRP():
    '''
    torch implementation of LRP
    http://www.heatmapping.org/tutorial/
    :param rule:str: name of used rule
    '''

    def __init__(self, model, rule, input_lowest=-1, input_highest=1):
        self.model = copy.deepcopy(model)
        self.model = self.model.eval()
        self.model = utils.redefine_nn(self.model, rule=rule, 
                input_lowest=input_lowest, 
                input_highest=input_highest) #redefine each layer(module) of model, to set custom autograd func
        self.output = None

    def forward(self, input_):
        '''
        performe usual forward pass
        '''
        self.local_input = input_.clone().detach()
        self.local_input.requires_grad_(True)
        output = self.model(self.local_input)
        return output

    def relprop(self, input_, R=None):
        '''
        call as default object call
        first perform forward pass, second relevance propagation
        '''
        #assert self.output is not None, 'First performe forward pass'
        output = self.forward(input_) #after this step object will have local_input attribute, which will request gradient
        if R is None: #if input R (relevance) is None select max logit
            R = (output == output.max()).float()
        output.backward(R, retain_graph=True)
        C = self.local_input.grad.clone().detach()
        assert C is not None, 'obtained relevance is None'
        self.local_input.grad = None
        R = C#*input_.clone().detach()
        return R

    __call__ = relprop


