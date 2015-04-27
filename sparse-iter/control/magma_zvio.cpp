/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

// includes, project
#include "common_magma.h"
#include "magmasparse_z.h"
#include "magma.h"
#include "mmio.h"



/**
    Purpose
    -------

    Visualizes part of a vector of type magma_z_matrix.
    With input vector x , offset, visulen, the entries 
    offset - (offset +  visulen) of x are visualized.

    Arguments
    ---------

    @param[in]
    x           magma_z_matrix
                vector to visualize

    @param[in]
    offset      magma_int_t
                start inex of visualization

    @param[in]
    visulen     magma_int_t
                number of entries to visualize       

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zprint_vector(
    magma_z_matrix x, 
    magma_int_t offset, 
    magma_int_t  visulen,
    magma_queue_t queue )
{
    magma_int_t stat_dev=0;

    //**************************************************************
    #define COMPLEX
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    
    #ifdef COMPLEX
    #define magma_zprintval( tmp )       {                                  \
        if ( MAGMA_Z_EQUAL( tmp, c_zero )) {                                \
            printf( "   0.              \n" );                                \
        }                                                                   \
        else {                                                              \
            printf( " %8.4f+%8.4fi\n",                                        \
                    MAGMA_Z_REAL( tmp ), MAGMA_Z_IMAG( tmp ));              \
        }                                                                   \
    }
    #else
    #define magma_zprintval( tmp )       {                                  \
        if ( MAGMA_Z_EQUAL( tmp, c_zero )) {                                \
            printf( "   0.    \n" );                                          \
        }                                                                   \
        else {                                                              \
            printf( " %8.4f\n", MAGMA_Z_REAL( tmp ));                         \
        }                                                                   \
    }
    #endif
    //**************************************************************
    
    printf("visualize entries %d - %d of vector ", 
                    (int) offset, (int) (offset + visulen) );
    fflush(stdout);  
    if ( x.memory_location == Magma_CPU ) {
        printf("located on CPU:\n");
        for( magma_int_t i=offset; i<offset + visulen; i++ )
            magma_zprintval(x.val[i]);
    return MAGMA_SUCCESS;
    }
    else if ( x.memory_location == Magma_DEV ) {
        printf("located on DEV:\n");
        magma_z_matrix y;
        stat_dev += magma_zmtransfer( x, &y, Magma_DEV, Magma_CPU, queue );
        for( magma_int_t i=offset; i<offset +  visulen; i++ )
            magma_zprintval(y.val[i]);
        magma_free_cpu(y.val);
        if( stat_dev !=0 ){
            return stat_dev;
        }
        return MAGMA_SUCCESS;
    }
    return MAGMA_SUCCESS;
}   





/**
    Purpose
    -------

    Reads in a double vector of length "length".

    Arguments
    ---------

    @param[out]
    x           magma_z_matrix *
                vector to read in

    @param[in]
    length      magma_int_t
                length of vector
    @param[in]
    filename    char*
                file where vector is stored
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zvread(
    magma_z_matrix *x, 
    magma_int_t length,
    char * filename,
    magma_queue_t queue )
{
    magma_int_t stat = 0;
    
    x->memory_location = Magma_CPU;
    x->storage_type = Magma_DENSE;
    x->num_rows = length;
    x->num_cols = 1;
    x->major = MagmaColMajor;
    stat += magma_zmalloc_cpu( &x->val, length );
    if( stat !=0 ){
        magma_free_cpu( x->val ) ;
        return stat;
    }
    magma_int_t nnz=0, i=0;
    FILE *fid;
    
    fid = fopen(filename, "r");
    
    while( i<length )  // eof() is 'true' at the end of data
    {
        double VAL1;

        magmaDoubleComplex VAL;
        #define COMPLEX
        
        #ifdef COMPLEX
            double VAL2;
            fscanf(fid, " %lf %lf \n", &VAL1, &VAL2);
            VAL = MAGMA_Z_MAKE(VAL1, VAL2);
        #else
            fscanf(fid, " %lf \n", &VAL1);
            VAL = MAGMA_Z_MAKE(VAL1, 0.0);
        #endif
        
        if ( VAL != MAGMA_Z_ZERO )
            nnz++;
        x->val[i] = VAL;
        i++;
    }
    fclose(fid);
    
    x->nnz = nnz;
    return MAGMA_SUCCESS;
}   




/**
    Purpose
    -------

    Reads in a sparse vector-block stored in COO format.

    Arguments
    ---------

    @param[out]
    x           magma_z_matrix *
                vector to read in

    @param[in]
    filename    char*
                file where vector is stored
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zvspread(
    magma_z_matrix *x, 
    const char * filename,
    magma_queue_t queue )
{
    magma_z_matrix A,B;
    magma_int_t entry=0;
     //   char *vfilename[] = {"/mnt/sparse_matrices/mtx/rail_79841_B.mtx"};
    magma_z_csr_mtx( &A,  filename, queue  ); 
    magma_zmconvert( A, &B, Magma_CSR, Magma_DENSE, queue );
    magma_zvinit( x, Magma_CPU, A.num_cols, A.num_rows, MAGMA_Z_ZERO, queue );
    x->major = MagmaRowMajor;
    for(magma_int_t i=0; i<A.num_cols; i++) {
        for(magma_int_t j=0; j<A.num_rows; j++) {
            x->val[i*A.num_rows+j] = B.val[ i+j*A.num_cols ];
            entry++;     
        }
    }
    x->num_rows = A.num_rows;
    x->num_cols = A.num_cols;

    magma_zmfree( &A, queue );
    magma_zmfree( &B, queue );

    return MAGMA_SUCCESS;
}   


