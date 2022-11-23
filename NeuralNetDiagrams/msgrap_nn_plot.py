
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

# Author: James Guzman (Medel)
# MSGRAP Network Architecture

def block_Trans3Conv( name, botton, top, s_filer="", n_filer=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5, caption="" ):
    return [
        # like Tranpose Conv (Upsampling)
        to_UnPool(  name='unpool_{}'.format(name),    offset=offset,    to="({}-east)".format(botton),         width=1,              height=size[0],       depth=size[1], opacity=opacity, caption=caption ),    
        to_Conv(    name='cccr_{}'.format(name),       offset="(0,0,0)", to="(unpool_{}-east)".format(name),   s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_Conv(    name='ccr_{}'.format(name),       offset="(0,0,0)", to="(cccr_{}-east)".format(name),   s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),       
        to_Conv(    name='{}'.format(top),            offset="(0,0,0)", to="(ccr_{}-east)".format(name), s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_connection( 
            "{}".format( botton ), 
            "unpool_{}".format( name ) 
            )
    ]

def block_Trans2Conv( name, botton, top, s_filer="", n_filer=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5, caption="" ):
    return [
        # like Tranpose Conv (Upsampling)
        to_UnPool(  name='unpool_{}'.format(name),    offset=offset,    to="({}-east)".format(botton),         width=1,              height=size[0],       depth=size[1], opacity=opacity, caption=caption ),
        to_Conv(    name='ccr_{}'.format(name),       offset="(0,0,0)", to="(unpool_{}-east)".format(name),   s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),       
        to_Conv(    name='{}'.format(top),            offset="(0,0,0)", to="(ccr_{}-east)".format(name), s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_connection( 
            "{}".format( botton ), 
            "unpool_{}".format( name ) 
            )
    ]

arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),
    
    #input
    to_input( '../examples/msgrap/ucsf_breast_cancer_mri.jpg' ),
    # to_input( '../examples/fcn8s/cats.jpg' ),

    #block-001: \nConv + ReLU + GN + \n Channel Attention (Excite Squeeze)
    to_ConvConvRelu( name='ccr_b1', s_filer="", n_filer=(64,64), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40 , caption="" ),
    to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", width=1, height=32, depth=32, opacity=0.5),
    
    #block-002
    *block_2ConvPool( name='b2', botton='pool_b1', top='pool_b2', s_filer="", n_filer=128, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5 ),

    #block-003
    to_Conv( name='c_b3', s_filer="", n_filer=256, offset="(1,0,0)", to="(pool_b2-east)", width=4.5, height=25, depth=25, caption="" ),
    to_ConvConvRelu( name='ccr_b3', s_filer="", n_filer=(256,256), offset="(0,0,0)", to="(c_b3-east)", width=(4.5,4.5), height=25, depth=25, caption="Conv + ReLU + GN + Channel Attention" ),
    to_Pool(name="pool_b3", offset="(0,0,0)", to="(ccr_b3-east)", width=1, height=(25 - int(25/4)), depth=(25 - int(25/4)), opacity=0.5, caption=""),
    to_connection( "pool_b2", "ccr_b3"),

    # *block_2ConvPool( name='b3', botton='c_b3', top='pool_b3', s_filer="", n_filer=256, offset="(0,0,0)", size=(25,25,4.5), opacity=0.5 ),

    #block-004: to={where we are coming from}
    to_Conv( name='c_b4', s_filer="", n_filer=512, offset="(1,0,0)", to="(pool_b3-east)", width=5.5, height=16, depth=16, caption="" ),
    to_ConvConvRelu( name='ccr_b4', s_filer="", n_filer=(512,512), offset="(0,0,0)", to="(c_b4-east)", width=(5.5,5.5), height=16, depth=16, caption=" " ),
    to_Pool(name="pool_b4", offset="(0,0,0)", to="(ccr_b4-east)", width=1, height=(16 - int(16/4)), depth=(16 - int(16/4)), opacity=0.5),
    to_connection( "pool_b3", "ccr_b4"),

    # *block_2ConvPool( name='b4', botton='c_b4', top='pool_b4', s_filer="",  n_filer=512, offset="(0,0,0)", size=(16,16,5.5), opacity=0.5 ),

    #Bottleneck
    #block-005
    to_Conv( name='c_b5', s_filer="", n_filer=512, offset="(2,0,0)", to="(pool_b4-east)", width=8, height=8, depth=8, caption="" ),
    to_ConvConvRelu( name='ccr_b5', s_filer="", n_filer=(512,512), offset="(0,0,0)", to="(c_b5-east)", width=(8,8), height=8, depth=8, caption=""  ),
    to_connection( "pool_b4", "ccr_b5"),

    #Decoder
    *block_Trans3Conv( name="b6", botton="ccr_b5", top="end_b6", s_filer="", n_filer=512, offset="(2.1,0,0)", size=(16,16,5.0), opacity=0.5, caption="Transpose Conv" ),
    to_skip( of='ccr_b4', to='cccr_b6', pos=1.25),

    # *block_Unconv( name="b6", botton="ccr_b5", top='end_b6', s_filer="",  n_filer=512, offset="(2.1,0,0)", size=(16,16,5.0), opacity=0.5 ),
    # to_skip( of='ccr_b4', to='ccr_res_b6', pos=1.25),

   *block_Trans3Conv( name="b7", botton="end_b6", top="end_b7", s_filer="", n_filer=256, offset="(2.1,0,0)", size=(25,25,4.5), opacity=0.5 ),
    to_skip( of='ccr_b3', to='cccr_b7', pos=1.25),

    # *block_Unconv( name="b7", botton="end_b6", top='end_b7', s_filer="", n_filer=256, offset="(2.1,0,0)", size=(25,25,4.5), opacity=0.5 ),
    # to_skip( of='ccr_b3', to='ccr_res_b7', pos=1.25),    

    *block_Trans2Conv( name="b8", botton="end_b7", top="end_b8", s_filer="", n_filer=128, offset="(2.1,0,0)", size=(32,32,3.5), opacity=0.5 ),

    # *block_Unconv( name="b8", botton="end_b7", top='end_b8', s_filer="", n_filer=128, offset="(2.1,0,0)", size=(32,32,3.5), opacity=0.5 ),
    # to_skip( of='ccr_b2', to='ccr_res_b8', pos=1.25),    
    
    *block_Trans2Conv( name="b9", botton="end_b8", top="end_b9", s_filer="", n_filer=64, offset="(2.1,0,0)", size=(40,40,2.5), opacity=0.5 ),

    # *block_Unconv( name="b9", botton="end_b8", top='end_b9', s_filer="", n_filer=64,  offset="(2.1,0,0)", size=(40,40,2.5), opacity=0.5 ),
    # to_skip( of='ccr_b1', to='ccr_res_b9', pos=1.25),
    
    # it says softmax, but I am using it as soigmoid
    to_ConvSoftMax( name="sig1", s_filer=2, offset="(0.75,0,0)", to="(end_b9-east)", width=1, height=40, depth=40, caption="SIGMOID" ),
    to_connection( "end_b9", "sig1"),
     
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    
