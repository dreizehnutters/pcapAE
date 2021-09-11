import torch
import numpy as np
import copy

#def __getattr__(name):
#    '''
#    to get rule functions as rules module attribut
#    '''
#    if name in globals():
#        return globals()[name]
#    else:
#        return AttributeError

class Rules(object):

    all_rules = ['z_rule', 'z_plus', 'z_rule_no_bias', 'z_plus_no_bias', 'z_box_no_bias']

    def __init__(self, rule, lowest=-1, highest=1):
        '''
        rule (str) - name of rule, one of {}
        lowest (int) and highest (int) - bounds of input image values
        '''.format(self.all_rules)
        assert isinstance(rule, str), 'rule parameter should be of type str'
        assert rule in self.all_rules, 'Rule "{}" not implemented. Implemented rules {}'.format(rule, self.all_rules)
        assert lowest < highest
        self.rule = rule
        self.lowest = lowest
        self.highest = highest

    def __call__(self, *args):
        if self.rule  == 'z_rule':
            return self.z_rule(*args, keep_bias=True)
        if self.rule  == 'z_rule_no_bias':
            return self.z_rule(*args, keep_bias=False)
        elif self.rule == 'z_plus':
            return self.z_plus(*args, keep_bias=True)
        elif self.rule == 'z_plus_no_bias':
            return self.z_plus(*args, keep_bias=False)
        elif self.rule == 'z_box_no_bias':
            return self.z_box(*args, keep_bias=False, lowest=self.lowest, highest=self.highest)
        else:
            ValueError('Rule "{}" not implemented'.format(rule))

    @staticmethod
    def z_rule(func, input, R, func_args, keep_bias=False):
        '''
        func - is a default layer torch function to be called of forward pass
        '''
        func_args = copy.deepcopy(func_args)
        input.requires_grad_(True)
        if input.grad is not None: input.grad.zero_() #otherwise accumulation of gradient happening
        if func_args.get('bias', None) is not None:
            if not keep_bias:
                func_args['bias'] = None
        with torch.enable_grad():
            Z = func(input, **func_args)
            S = R /(Z + (Z==0).float()*np.finfo(np.float32).eps)
            Z.backward(S)
            assert input.grad is not None
            C = input.grad
            Ri = input * C
            print(Ri.sum())
        return Ri

    @staticmethod
    def z_plus(func, input, R, func_args, keep_bias=False):

        func_args = copy.deepcopy(func_args) #need to not change default parameters
        input.requires_grad_(True)
        if input.grad is not None: input.grad.zero_()
        if func_args.get('bias', None) is not None:
            if not keep_bias:
                func_args['bias'] = None
        if func_args.get('weight', None) is not None:
            func_args['weight'].clamp_(0, float('inf'))
        with torch.enable_grad():
            Z = func(input, **func_args)
            S = R /(Z + (Z==0).float()*np.finfo(np.float32).eps)
            Z.backward(S)
            assert input.grad is not None
            C = input.grad
            Ri = input * C
        return Ri

    @staticmethod
    def z_box(func, input, R, func_args, lowest, highest, keep_bias=False):
        '''
        if input constrained to bounds lowest and highest
        usually used as first layer,
        and input image is bound to -1 1 interval
        '''
        assert input.min() >= lowest
        assert input.max() <= highest
        ifunc_args = copy.deepcopy(func_args)
        nfunc_args = copy.deepcopy(func_args)
        nfunc_args['weight'].clamp_(-float('inf'), 0)
        pfunc_args = copy.deepcopy(func_args)
        pfunc_args['weight'].clamp_(0, float('inf'))
        if func_args.get('bias', None) is not None:
            if not keep_bias:
                ifunc_args['bias'] = None
                nfunc_args['bias'] = None
                pfunc_args['bias'] = None
        L = torch.zeros_like(input) + lowest
        H = torch.zeros_like(input) + highest
        L.requires_grad_(True)
        #if L.grad is not None: L.grad.zero_()
        L.retain_grad()
        H.requires_grad_(True)
        #if H.grad is not None: H.grad.zero_()
        H.retain_grad()
        input.requires_grad_(True)
        if input.grad is not None: input.grad.zero_()
        input.retain_grad()
        with torch.enable_grad():
            Z = func(input, **ifunc_args) - func(L, **pfunc_args) - func(H, **nfunc_args)
            S = R / (Z + (Z==0).float()*np.finfo(np.float32).eps)
            Z.backward(S)
            assert input.grad is not None
            assert L.grad is not None
            assert H.grad is not None
            import ipdb; ipdb.set_trace() # BREAKPOINT
            Ri = input * input.grad - L * L.grad - H * H.grad
        return Ri
#
#
#def z_epsilon_rule(module, input_, R, keep_bias=True):
#    '''
#    '''
#    pself = module
#    if hasattr(pself, 'bias'):
#        if not keep_bias:
#            pself.bias.data.zero_()
#   # if hasattr(pself, 'weight'):
#   #     pself.weight.data.clamp_(0, float('inf'))
#    Z = pself(input_)
#    S = R / (Z + ((Z>=0).float()*2-1)*np.finfo(np.float32).eps)
#    Z.backward(S)
#    C = input_.grad
#    R = input_ * C
#    return R
#
#
#def alfa_beta_rule(module, input_, R, alfa=2, keep_bias=False):
#    '''
#    General rule, alfa = 1 is z_plus_rule case. 
#    '''
#    #TODO: imlementation https://github.com/albermax/innvestigate/blob/accbb99d0da2eb47f22e3d04563c8964e2b1ad90/innvestigate/analyzer/relevance_based/relevance_rule.py#L212
#    #not the same as in http://www.heatmapping.org/tutorial/
#    assert alfa >= 1, 'alfa should be >=1, but got {}'.format(alfa)
#    #assert beta >= 1, 'beta should be >=0, but got {}'.format(beta)
#    #assert alfa-beta == 1, 'alfa-beta should be equal to 1, but got {}'.format(alfa-beta)#TODO:why alfa-beta=1 not alfa+beta = 1?
#    nself = copy.deepcopy(module)
#    if hasattr(nself, 'weight'):
#        nself.weight.data.clamp_(-float('inf'), -np.finfo(np.float32).eps)
#    pself = copy.deepcopy(module)
#    if hasattr(pself, 'weight'):
#        pself.weight.data.clamp_(np.finfo(np.float32).eps, float('inf'))
#    if hasattr(pself, 'bias'):
#        if not keep_bias:
#            pself.bias.data.zero_()
#            nself.bias.data.zero_()
#    inputA_ = input_ + np.finfo(np.float32).eps
#    inputB_ = input_ + np.finfo(np.float32).eps
#    inputA_.requires_grad_(True)
#    inputA_.retain_grad()
#    inputB_.requires_grad_(True)
#    inputB_.retain_grad()
#    ZA = pself(inputA_)
#    SA = alfa*R/ZA
#    ZA.backward(SA)
#    ZB = nself(inputB_)
#    SB = -1*(alfa-1)*R/ZB
#    ZB.backward(SB)
#    Ri = input_*(inputA_.grad + inputB_.grad)
#    return Ri
#
#
#
#def w2_rule(module, input_, R):
#    '''
#    if input is unconstrained
#    '''
#    pself = module
#    if hasattr(pself, 'bias'):
#        pself.bias.data.zero_()
#    if hasattr(pself, 'weight'):
#        pself.weight.data = pself.weight.data ** 2
#    Z = pself(input_)
#    S = R / pself(torch.ones_like(input_))
#    Z.backward(S)
#    R = input_.grad
#    return R
