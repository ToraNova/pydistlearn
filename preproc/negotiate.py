# Declares and defines the methods that allows
# the central and clients to commit negotiations
# Negotations is done before the distributed learning
# is conducted to allow all parties to sync up and agree
# on the training size

# This class act as a message packet which can be pickled and unpickled
# to allow sending through sockets
class NegForm:

    # this is the primary data for the form
    # one can use json dumps to serialize this
    # send it over a socket and recreate a negform
    # on the other end using the copy constructor
    negform_primary = {
            "esize":None,   # entity/entries   size
            "fsize":None,   # feature/target   size
            "bsize":None,   # batch/splitting  size
            "dflag":False,  # dflag check if it has the target
            "rrlambda":None # rrlambda is the ridge regression's hyperparam
            }

    def __init__(self, constr_param=None):
        '''initializes the Negform for the ConceptDonor
        itself, this allows sending of params to the
        central'''
        if(constr_param == None):
            # this is the central
            pass

        elif( type(negform_primary) == dict ):
            # copy constructor
            negform_primary = constr_param

        else:
            # initializes the negform
            self.primary['esize'] = constr_param.__mDmat.shape[0]
            self.primary['fsize'] = constr_param.__mDmat.shape[1]
            self.primary['dflag'] = constr_param.hasTarget 

    dsform = '''es/fs/bs : (%d/%d/%d) [df?:%s]\nrrlambda : %.2f'''
    def display():
        '''displays the content of the negform'''
        print(dsform % (
            self.primary["esize"],
            self.primary["fsize"],
            self.primary["bsize"],
            self.primary["dflag"],
            self.primary["rrlambda"]))
        



