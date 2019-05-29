# Rewriter - a module to aid csv data preprocessing ('rewrites')
import csv, numpy, pandas
from sklearn.preprocessing import   StandardScaler, LabelEncoder,\
                                    OrdinalEncoder, MinMaxScaler
from pyioneer.support import Pam

class NeoNPDController(Pam):
    '''CSV rewriter, allows csv rewriting - preprocessing,
    built with sklearn / pandas and numpy
    Update - now this replaces the old NPD controller - 
    renamed to NeoNPDcontroller'''
    fname = 'rewriter_default.csv'
    _constant_axis_COLS = 1
    _constant_axis_ROWS = 0


    def __init__(self, filename, index_col=0, adelimiter=',',verbose=False,debug=False):
        super().__init__(verbose=verbose,debug=debug)
        self.df = pandas.read_csv( filename, index_col=index_col , delimiter=adelimiter )
        self.bu = self.df #backup copy
        self.fname = filename
        # define the standard cat-num methods
        self.hlist = self.df.columns.values.tolist()

    def getsize(self):
        return self.df.shape

    def head(self,n=10):
        '''prints the first n'''
        self.verbose("Printing first {} rows".format(n))
        print( self.df.head(n) )

    def tail(self,n=10):
        '''prints the last n'''
        self.verbose("Printing last {} rows".format(n))
        print( self.df.tail(n) )

    def fields(self):
        '''prints the column headers'''
        print( self.df.columns.values.tolist() )

    def emptydisp(self):
        '''prints all rows with NaN'''
        self.verbose("Printing rows with NaN")
        empval = numpy.where( pandas.isnull(self.df))
        print( self.df.iloc[ empval[0] ])

    def emptyrows(self):
        '''prints the empty rows and what fields is empty'''
        self.verbose("Printing NaN rows")
        empval =  self.df.isnull().any(axis=self._constant_axis_COLS)
        print(self.df[empval])
        return empval

    def emptycols(self):
        '''prints the empty cols and what fields is empty'''
        self.verbose("Printing NaN columns")
        empval =  self.df.isnull().any(axis=self._constant_axis_ROWS)
        print(empval)
        return empval

    ############################################################################
    # Used by pandframe.py rather than as a preprocessing script
    ############################################################################
    def obtrain(self,n, targetname=None ):
        train,test = NeoNPDController.dframe_split(self.df, n )
        if(targetname is not None):
            train_y = train[targetname]
            test_y = test[targetname]
            train_x = train.drop( [targetname], axis = self._constant_axis_COLS )
            test_x = test.drop( [targetname], axis = self._constant_axis_COLS )
            return train_x.values,test_x.values, train_y.values,test_y.values
        else:
            return train.values, test.values
        
    def edtrain(self):
        self.df.join( self.target )

    @staticmethod
    def horizontal_split(dframe, n):
        '''splits the internal array horizontally returning returning the 
        train and test dataframes'''
        dsplit = numpy.split( dframe, [n], axis = NeoNPDController._constant_axis_ROWS )
        return dsplit[1].values, dsplit[0].values

    @staticmethod
    def dframe_split(dframe,n):
        '''like horizontal split, except returns dframes intead of ndarrays'''
        dsplit = numpy.split( dframe, [n], axis = NeoNPDController._constant_axis_ROWS )
        return dsplit[1], dsplit[0]

    ############################################################################

    def selectcolumn(self, cols, met, arglist):
        '''select a column or a set of columns to be rewritten,
        the unselected ones are automatically ignored (this is like
        a reverse ignorecolumn
        @cols - specify which columns to select
        @dummy - met args, Not used for this preprocessing method'''
        if( isinstance(cols,list)):
            #self.df = self.df.drop( self.df.columns[cols], axis=self._constant_axis_COLS)
            self.df = self.df.filter( cols, axis= self._constant_axis_COLS)
            #for c in cols:
            #    del self.df[c]
        else:
            self.df = self.df.filter( [cols], axis =self._constant_axis_COLS)
            #del self.df[c]
        return True


    def ignorecolumn(self, cols, met, arglist):
        '''ignore a singular or a set of columns from being rewritten (
        exclusion of certain features)
        @cols - specify which columns to ignore,
        @dummy - met args, Not used for this preprocessing method'''
        if( isinstance(cols,list)):
            #self.df = self.df.drop( self.df.columns[cols], axis=self._constant_axis_COLS)
            self.df.drop( cols, axis= self._constant_axis_COLS, inplace=True)
            #for c in cols:
            #    del self.df[c]
        else:
            self.df.drop( [cols], axis =self._constant_axis_COLS, inplace=True)
            #del self.df[c]
        return True

    def printcertain(self, cols):
        print(self.df[cols])

    def fill_empty(self, cols, met , arglist):
        '''perform filling based on the selected method (with value or uses mode)
        fill the empty columns with the mode of that column
        e.g fillempty_withmode( ['Col1','Col2','Col7'] )
        fill Col1, Col2 and Col7's empty (NaN values) with the mode of the column
        if cols is none (left empty), then applies to all
        @cols - specify which column to fill ( a list )
        @met - mode/value specify how to fill
        @args - arg[0] - value : fill to this value
        '''
        #TODO: Please check out the .fillna method
        # e.g : df[c].fillna( df[c].mode()[0],inplace=True)
        if( cols is None):
            #fill all
            cols = self.hlist
        elif( not isinstance(cols, list)):
            cols = [cols] #convert to list
        ## WIP FILLNA METHOD
        for c in cols:
            if(met=="mode"):
                self.df[c].fillna( self.df[c].mode().values[0], inplace=True )
            elif(met=="value"):
                self.df[c].fillna( arglist[0], inplace=True )
            elif(met=="mean"):
                self.df[c].fillna( self.df[c].mean() , inplace=True )
            else:
                self.error("Error! Invalid method/args specified for fill_empty()")
                return False
        return True

        # OLD METHOD
        #fill specified
        #for c in cols:
        #    #generate replacer
        #    if(met=="mode"):
        #        replacer = self.df[c].mode().values[0]
        #    elif(met=="value"):
        #        replacer = float(arglist[0])
        #    elif(met=="mean"):
        #        replacer = self.df[c].mean().values[0]
        #    else:
        #        #error
        #        self.error("Error! Invalid method/args specified for fill_empty()")
        #        return False
        #    #check if empty rows exist at all
        #    empval =  numpy.where(self.df.isnull())
        #    cn = self.df.columns.get_loc(c)
        #    if(len(empval[0])>=1):
        #        for n in empval[0]:
        #            self.df.iloc[ n,cn ] = replacer
        #success
        return True

    def apply_mapper(self, cols ,mets ,  arglist ):
        '''apply mapper to the columns
        example :
        apply_mapper( ['Age Group','Ethnicity'],'labelencoder' )
        applies LabelEncoder to AgeGroup and Ethnicity.
        mets can be just 'LabelEncoder' and it will be assigned to all the columns listed 
        by cols. cols can be none and it will apply to all the column
        @cols - specify which column to apply the mapper
        @met - which method (labelencoder, onehotencoder, standardscaler, ordinalencoder)
        '''
        # mets is singular, apply it to all
        if( cols is not None):
            #apply to specific
            if( not isinstance( cols,list )):
                cols = [cols] #convert to list
            return self.lkup[mets](self,cols,arglist)
        else:
            #apply to all
            return self.lkup[mets](self,self.hlist,arglist)

    def _apply_labelencoder(self, cols, arglist):
        '''applies the labelencoder over the columns.'''
        ec = LabelEncoder()
        for c in cols:
            ec.fit( self.df[c] )
            self.df[c] = lec.transform( self.df[c] )
        return True

    def _apply_standardscaler(self,cols, arglist):
        '''applies the standardscaler such that the transform will have a
        std. dev. of 1 and a mean of 0. cols should ideally be a list'''
        sc = StandardScaler()
        self.df[cols] = sc.fit_transform( self.df[cols] )
        return True

    def _apply_minmaxscaler(self,cols, arglist):
        '''applies the minmaxscaler such that the feature is scaled
        to within range specified in args. see sklearn.preprocessing.MinMaxScaler'''
        sc = MinMaxScaler((args[0],args[1])) #args0 - min, args1 - max
        # by default minmax scaler uses 0 as min and 1 as max
        self.df[cols] = sc.fit_transform( self.df[cols] )
        return True

    def _apply_onehotencoder(self,cols, arglist):
        '''cols should be a list !, applies one hot encoder to the columns.
        update1: now cols can be singular, the function auto converts it to a list'''
        dummies = pandas.get_dummies( self.df.filter( cols, axis= self._constant_axis_COLS))
        self.df.drop( cols, axis= self._constant_axis_COLS, inplace=True)
        self.df = self.df.join( dummies )
        return True

    def rearrange_cols(self, target, mets, arglist ):
        '''rearrange the columns. move target end of the dataframe
        the last column -- useful to moving the target to the end for training'''
        # TODO: allow target to be moved to specified index
        excludelist = self.df.columns.values.tolist()
        if( target in excludelist ):
            if( isinstance(target,list) ):
                self.error(target,"cannot be a list in rearrange operation")
                return False
            ind_tar = excludelist.index(target)
            del excludelist[ind_tar]
            self.df = self.df[ excludelist + [target] ]
            return True
        else:
            self.error(target,"not a column in the dataframe!")
            return False

    def rewrite( self, filename = fname ,aindex=False):
        '''rewrites the dataframe to a csv file'''
        self.df.to_csv( filename , index=aindex)

    def preproc(self, procedure, col, mets, arglist):
        '''easy caller. use this to call the preprocessing methods'''
        '''use None as placeholder for mets if not applicable'''
        res =  self.proc[procedure](self,col,mets,arglist)
        self.hlist = self.df.columns.values.tolist()
        return res

    proc = {
        "preproc": apply_mapper,
        "ignore": ignorecolumn,
        "fill": fill_empty,
        "select": selectcolumn,
        "moveback": rearrange_cols
        }

    lkup = {
        "labelencoder": _apply_labelencoder,
        "onehotencoder": _apply_onehotencoder,
        "standardscaler":_apply_standardscaler,
        "minmaxscaler":_apply_minmaxscaler
        }

