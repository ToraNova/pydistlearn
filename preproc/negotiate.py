# Declares and defines the methods that allows
# the central and clients to commit negotiations
# Negotations is done before the distributed learning
# is conducted to allow all parties to sync up and agree
# on the training size

# This class act as a message packet which can be pickled and unpickled
# to allow sending through sockets
import pyioneer.support.gpam as gpam
class NegForm:

    # this is the primary data for the form
    # one can use json dumps to serialize this
    # send it over a socket and recreate a negform
    # on the other end using the copy constructor
    primary = {
            "esize":-1,   # entity/entries   size
            "fsize":-1,   # feature/target   size
            "bsize":-1,   # batch/splitting  size
            "dflag":False,  # dflag check if it has the target
            "rrlambda":-1 # rrlambda is the ridge regression's hyperparam
            }

    def __init__(self, constr_param=None):
        '''initializes the Negform for the ConceptDonor
        itself, this allows sending of params to the
        central'''
        if(constr_param == None):
            # this is the central
            gpam.debug("Negform -- central constructor")
            pass

        elif( type(constr_param) == dict ):
            # copy constructor
            gpam.debug("Negform -- copy constructor",constr_param)
            self.primary = constr_param

        else:
            # initializes the negform
            gpam.debug("Negform -- donor constructor")
            sz = constr_param.sizeof_internals()
            self.primary['esize'] = sz.get('data')[0]
            self.primary['fsize'] = sz.get('data')[1]
            self.primary['dflag'] = constr_param.hasTarget 

    def isfilled(self):
        '''returns true if the negform is considered to be synced'''
        if( self.primary['bsize'] != -1 and self.primary['rrlambda'] != -1):
            return True
        else:
            return False

    dsform = '''es/fs/bs : (%d/%d/%d) [df?:%s]\nrrlambda : %.2f'''
    def display(self):
        '''displays the content of the negform'''
        print(self.dsform % (
            self.primary["esize"],
            self.primary["fsize"],
            self.primary["bsize"],
            self.primary["dflag"],
            self.primary["rrlambda"]))
        



