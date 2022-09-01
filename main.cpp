#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "define2.h"
#include "sample_print.c"
#include "data_generation.c"
#include "clsvm.c"
#include "fairness.c"
#include <assert.h>
/*******************
 * make clean
 * make all
 * make execute
*******************/
 

//////////////////////// Iulia: added
double *distance;
////////////////////////

   static int
setproblemdata_SVM (CPXENVptr *env, CPXLPptr *lp, int fila,int testsize,double Ci, double pi, int di,int M,double **sample,
					   int *lab,double **tes_sample, int *tes_label, double **val_sample,int *val_label,double **x_p,
					   double *objval,int *statusCR,int solve, double *total_time,int *g, double Delta, double beta);

static bool loadValues(FILE * file, int * FILA, double * C, int * a, double * p, int *d, int *M, int *delta2, double *beta,
					   double *mu1, double *sigma1, double *mu2, double *sigma2, int *outlier, double *tolerance,double *W,
					   int *nodelim, double *tilim);

int main (int argc,char *argv[])
 {

	FILE * aux= NULL;
	FILE *input = fopen(argv[1],"r"); //config.txt
	int  FILA, d, M, a, i,nodelimaux,status,instance,type,solstat,statusCR,num_continuas,num_binarias,num_categoricas,var_sensible,cat_protected_single,delta2;
	double C, mu1, sigma1, mu2, sigma2,W,timelimit,p,tole,total_time,lbound,gap,db,objval,CRobjvalue,timerelx,eqotrai_svm,eqotes_svm,eqoval_svm,demopar_trai,demopar_tes,demopar_val,beta;
	CPXENVptr     env = NULL;
	CPXLPptr      lp = NULL;
	time_t start, end;
	double  **testing_sample = 0,**big_sample=0;
	double **training_sample=0;
	double **validation_sample=0;
	int		*label = 0;
	int *training_label = 0,*cat_protected=0;
	int *validation_label=0,*v_eqop_tr=0,*v_eqop_test=0,*v_eqop_val=0,*acum_K=0,*K_j=0,*big_label=0;
	char *zctype=0;
	double *x_p=0;
	double *CRsolution=0,acctes_svm,accval_svm,acctrai_svm,UNF_EqOpptrai_svm,UNF_EqOpptes_svm,UNF_EqOppval_svm,UNF_PEtrai_svm,UNF_PEtes_svm,UNF_PEval_svm;

	//Iulia: added for #P, #N
	int *g;
	/////////////////////////

	if(input == NULL){
	printf("Can not open file input. Check config.txt path");
	}

	aux = fopen("databrooks.txt","w");
	fclose(aux);
	aux = fopen("SVMsolution.txt","w");
	fclose(aux);
	aux = fopen("metrics.txt","w");
	fclose(aux);


	
	/* Initialize the CPLEX environment */
	env = CPXopenCPLEX (&status);
	if (status){
		printf("Can not open CPLEX");
	}


	//Los bucles hay que ponerlos despu�s de esto, para que no renueve siempre la semilla.
	//The loops must be placed after this, so that it does not always renew the seed.
	instance=0;

	/*
	* Iulia: input - argv[1] - config.txt
	* ex: FILA500	C500000000	a1000	p2	d59	M1000	delta2-0	beta0.25
	* mu1-0	sigma1-2	mu1-7	sigma2-2	type12	tole95	W0	nodelimax100000000	timelimit300
	*/

	/* 
	Iulia: changeable: 13 times 10 instances in config.txt file
	*/

	while(loadValues(input, &FILA, &C, &a, &p, &d, &M, &delta2, &beta, &mu1, &sigma1, &mu2, &sigma2, &type, &tole, &W ,&nodelimaux, &timelimit))
	{

	//ADDED 
	timelimit = 600;
	C = C/5;
	printf("\n\n===========Processing  %d %d %f %d %f %d %d============\n\n", instance+1, FILA, C, a,p,d, M);
	fill_data1(type,&num_continuas,&num_categoricas,&num_binarias); //Iulia: setting numbers
	int numcols= d+1+3*FILA; //Iulia: 5/07: changed from d+1+FILA for beta & z
	int numrows= FILA;
	int testsize=(a-FILA)/2; 

/// iulia: added
double Delta = 0.001;
beta = 0.25; 
printf("\n Delta = %f :: beta = %f :: FILA %d dim %d\n :: time_lim %f", Delta, beta, FILA, d, timelimit); 
////////////////
	g = vectorint(a);
	//ojo controlar cada cuando queremos que actualice la semilla
	//eye control every time we want to update the seed
	if (instance==10){
	instance=0;
	}
	instance=instance+1;
	//status=CPXsetintparam(env,CPX_PARAM_HEURFREQ,1);
	//status=CPXsetintparam(env,CPX_PARAM_NODESEL,0);
	//status=CPXsetintparam(env,CPX_PARAM_VARSEL,3);
	status= CPXsetdblparam (env, CPX_PARAM_TILIM, timelimit);
	status=CPXsetintparam (env,CPX_PARAM_ADVIND,1); 
	status= CPXsetintparam(env, CPX_PARAM_NODELIM,nodelimaux);
	status= CPXsetdblparam (env, CPX_PARAM_TRELIM, 1e10);
	status = CPXsetintparam (env, CPX_PARAM_NODEFILEIND, 2);
	status = CPXsetintparam (env, CPX_PARAM_MIPINTERVAL, 1);
	//status = CPXsetintparam (env, CPX_PARAM_MIRCUTS,-1);
	status = CPXsetdblparam (env,CPX_PARAM_WORKMEM, 56.0);
		
		
	aux = fopen("databrooks.txt","a");
	if (type==1){
	fprintf(aux,"%d\t%d\t%f\t%d\t%d\t%f\tA\t%d\t",instance, FILA,C, d, M, timelimit,nodelimaux);
	}
	else if (type==2){
	fprintf(aux,"%d\t%d\t%f\t%d\t%d\t%f\tB\t%d\t",instance, FILA,C, d, M, timelimit,nodelimaux);
	}
	else if (type==3){
	fprintf(aux,"%d\t%d\t%f\t%d\t%d\t%f\tC\t%d\t",instance, FILA,C, d, M, timelimit,nodelimaux);
	}
	else {
	fprintf(aux,"%d\t%d\t%f\t%d\t%d\t%f\t%d\t",instance, FILA,C, d, M, beta,delta2);
	}

	fclose(aux);


	aux = fopen("metrics.txt","a");
	fprintf(aux,"%d\t%10.2f\t",instance,beta);
	fclose(aux);


	//generacion de la muestra
	testing_sample		= matrix2(d,testsize);
	training_sample		= matrix2(d,FILA);
	validation_sample	= matrix2(d,testsize);
	label				= vectorint(testsize);
	training_label		= vectorint(FILA);
	validation_label	= vectorint(testsize);
	big_sample			= matrix2(d,a);
	big_label			= vectorint(a);
	CRsolution			= vector(numcols);
	zctype				= vectorchar(numcols);
	x_p					= vector(numcols);
	//Categorical
	K_j					=vectorint(num_categoricas);
	acum_K				=vectorint(num_categoricas);
	//Fairness 
	v_eqop_tr			=vectorint(FILA);
	v_eqop_test			=vectorint(testsize);
	v_eqop_val			=vectorint(testsize);
	

	fill_data2(type, K_j,acum_K);  //%%% setting numbers

		if (type==12){
		var_sensible=6;
		cat_protected=vectorint(K_j[var_sensible-1]);
		

		cat_protected[1]=1;
		cat_protected_single=2;
	}

	//Base Adult: sensible feature race, protected category black
 	if (type==5){
		var_sensible=9;
		cat_protected		=vectorint(K_j[var_sensible-1]);
		
		cat_protected_single=5;
		cat_protected[cat_protected_single-1]=1;
	
	} 

	//Base Adult: sensible feature relationship, protected category wife
/* 	if (type==5){
		var_sensible=8;
		cat_protected		=vectorint(K_j[var_sensible-1]);
		
		cat_protected_single=1;
		cat_protected[cat_protected_single-1]=1;
	
	} 

	//Base Adult: sensible feature relationship, protected category Own-child
 	if (type==5){
		var_sensible=8;
		cat_protected		=vectorint(K_j[var_sensible-1]);
		
		cat_protected_single=2;
		cat_protected[cat_protected_single-1]=1;
	
	} */

	//COMPAS
	if (type==13){
		var_sensible=1;
		cat_protected=vectorint(K_j[var_sensible-1]);
		
		cat_protected_single=2;
		cat_protected[cat_protected_single-1]=1;
	
	}

	//COMPAS_ext
	if (type==14){
		var_sensible=1;
		cat_protected=vectorint(K_j[var_sensible-1]);
		
		cat_protected_single=2;
		cat_protected[cat_protected_single-1]=1;
	
	}


//type = 12 => German(nb_instance).txt

	if (type==1){
		samplea(FILA,d,a, p, mu1, sigma1, mu2,sigma2,testing_sample,training_sample,validation_sample, label, training_label, validation_label);
	}
			
	else if(type==2){
		sampleb(FILA,d,a, p, mu1, sigma1, mu2,sigma2,testing_sample,training_sample,validation_sample, label, training_label, validation_label);

	}
	else if(type==3){
		samplec(FILA,d,a ,p, mu1, sigma1, mu2,sigma2,testing_sample,training_sample,validation_sample, label, training_label,validation_label);
	}
	else if(type>=4){
		sampled_CLSVM(FILA,d,a,type,instance,testing_sample,training_sample,validation_sample, label, training_label, validation_label,big_sample,big_label);
	}
	else {
		printf("Wrong type of outliers introduced");
		goto TERMINATE;
	}

	//Iulia: added here for NEW constraints (N,P)
	is_protected(FILA, (a-FILA)/2, training_sample, testing_sample,validation_sample,v_eqop_tr, v_eqop_test, v_eqop_val, var_sensible, cat_protected,num_continuas, acum_K, K_j);

	for(i=0;i<FILA;i++){
		g[i] = v_eqop_tr[i];
	}
	
	for(i=0;i<(a-FILA)/2;i++){
		g[FILA+i] = v_eqop_test[i];
	}

	for(i=0;i<(a-FILA)/2;i++){
		g[FILA+(a-FILA)/2+i] = v_eqop_val[i];
	}

	/* If an error occurs, the status value indicates the reason for
		failure.  A call to CPXgeterrorstring will produce the text of
		the error message.  Note that CPXopenCPLEX produces no output,
		so the only way to see the cause of the error is to use
		CPXgeterrorstring.  For other CPLEX routines, the errors will
		be seen if the CPX_PARAM_SCRIND indicator is set to CPX_ON.  */

	if ( env == NULL ) {
		char  errmsg[1024];
		fprintf (stderr, "Could not open CPLEX environment.\n");
		CPXgeterrorstring (env, status, errmsg);
		fprintf (stderr, "%s", errmsg);
		goto TERMINATE;
	}
			
		
		/* Turn on output to the screen */
	status = CPXsetintparam (env, CPX_PARAM_SCRIND, CPX_OFF);
	if ( status ) {
		fprintf (stderr,
				"Failure to turn on screen indicator, error %d.\n", status);
		goto TERMINATE;
	}

 	//Iulia: added g for NEW constraints.

	status = setproblemdata_SVM (&env, &lp, FILA,testsize, C, p, d,M,training_sample, training_label,testing_sample,label, validation_sample,validation_label,&CRsolution,&CRobjvalue,&statusCR,0,&timerelx,g, Delta,beta);
	if ( status ) {
	fprintf (stderr, "Failed to build brooks relaxation.\n");
	goto TERMINATE;
	}
	/* Finally, write a copy of the problem to a file. */
	status = CPXwriteprob (env, lp, "SVM.lp", NULL);
	if (status) {
	fprintf (stderr, "Failed to write LP to disk.\n");
	goto TERMINATE;
	}
	
	/* Turn on output to the screen */
	status = CPXsetintparam (env, CPX_PARAM_SCRIND, CPX_OFF);
	if ( status ) {
		fprintf (stderr,
				"Failure to turn on screen indicator, error %d.\n", status);
		goto TERMINATE;
	}

	//double start1, end1;
	
	//start1=CPU_TIME;

	start = clock();
		printf("\n start time %time_t\n", start);
	/* Optimize the problem and obtain solution. */
	status = CPXmipopt (env, lp);
	if ( status ) {
	printf("\nSTATUS non-zero %d", solstat);
	// https://www.ibm.com/docs/en/icos/20.1.0?topic=micclcarm-solution-status-codes-by-number-in-cplex-callable-library-c-api
	solstat = CPXgetstat (env, lp);
	fprintf (stderr, "Failed to optimize MIQP.\n");
	goto TERMINATE;
	}

	end = clock();
	//end1 = CPU_TIME;
	//printf("\n end time %time_t\n", end);
	//printf ("\n time.h %f\n", end1-start1);
total_time = (double)( end - start )/(double)CLOCKS_PER_SEC;
	printf("\n TOTAL TIME: %f s\n", total_time);

	solstat = CPXgetstat (env, lp);
	printf("\nSTATUS %d", solstat);

	if (solstat!=101 && solstat!=102 && solstat!=105 && solstat!=107 && solstat!=106 && solstat!=108){
	goto TERMINATE;
	}

	if (solstat!=106 && solstat!=108){

	status = CPXgetobjval (env, lp, &objval);
	if ( status ) {
	fprintf (stderr,"No MIQP objective value available.  Exiting...\n");
	goto TERMINATE;
	}
	status = CPXgetbestobjval (env, lp, &db);

	gap= abs(objval-db)*100/abs(objval);

	status = CPXgetx (env, lp, x_p, 0, numcols-1);
	if ( status ) {
	fprintf (stderr, "Failed to get optimal integer x.\n");
	goto TERMINATE;
	}

    for (int j = 0; j < numcols-FILA; j++) {
      printf ( "\nColumn %d:  Value = %17.10g", j, x_p[j]);
    }
    printf("\n");
    for (int j = numcols-FILA; j < numcols; j++) {
      printf ( "\n z[%d] = %17.10g", j-numcols+FILA, x_p[j]);
    }
////////////////////////////////////////////////////////////////////////////////

	accval_svm= Accuracy (testsize, d, x_p, validation_sample, validation_label);
	acctrai_svm= Accuracy (FILA, d, x_p, training_sample, training_label);
	acctes_svm= Accuracy (testsize, d, x_p, testing_sample, label);
	//vector_equalopportunity(FILA,testsize,training_sample, training_label,testing_sample, label,validation_sample, validation_label,v_eqop_tr,v_eqop_test,v_eqop_val,var_sensible,cat_protected,num_continuas, acum_K);
	is_protected( FILA, testsize,training_sample, testing_sample,validation_sample,v_eqop_tr, v_eqop_test, v_eqop_val, var_sensible, cat_protected,num_continuas, acum_K, K_j);

	eqotrai_svm= equal_opportunity(FILA, d, x_p, training_sample, training_label,var_sensible,cat_protected,num_continuas, acum_K,K_j,v_eqop_tr);
	eqotes_svm= equal_opportunity(testsize, d, x_p, testing_sample, label,var_sensible,cat_protected,num_continuas, acum_K,K_j,v_eqop_test);
	eqoval_svm= equal_opportunity(testsize, d, x_p, validation_sample, validation_label,var_sensible,cat_protected,num_continuas, acum_K,K_j,v_eqop_val);
	
	UNF_EqOpptrai_svm= UNF_EqOpp(FILA, d, x_p, training_sample, training_label,var_sensible,cat_protected,num_continuas, acum_K,K_j,v_eqop_tr);
	UNF_EqOpptes_svm= UNF_EqOpp(testsize, d, x_p, testing_sample, label,var_sensible,cat_protected,num_continuas, acum_K,K_j,v_eqop_test);
	UNF_EqOppval_svm= UNF_EqOpp(testsize, d, x_p, validation_sample, validation_label,var_sensible,cat_protected,num_continuas, acum_K,K_j,v_eqop_val);

	UNF_PEtrai_svm= UNF_PE(FILA, d, x_p, training_sample, training_label,var_sensible,cat_protected,num_continuas, acum_K,K_j,v_eqop_tr);
	UNF_PEtes_svm= UNF_PE(testsize, d, x_p, testing_sample, label,var_sensible,cat_protected,num_continuas, acum_K,K_j,v_eqop_test);
	UNF_PEval_svm= UNF_PE(testsize, d, x_p, validation_sample, validation_label,var_sensible,cat_protected,num_continuas, acum_K,K_j,v_eqop_val);
	



	lbound=0;

	aux = fopen("databrooks.txt","a");
	printing_data_SVM (aux,  d, total_time, objval,solstat, 0.0,acctrai_svm,acctes_svm,accval_svm,eqotrai_svm,eqotes_svm,eqoval_svm,x_p,0.0,acum_K, K_j);
	fclose(aux);
	aux = fopen("databrooks.txt","a");
	fprintf(aux,"\n");
	fclose(aux);

	aux = fopen("SVMsolution.txt","a");
		for (i=0;i<=d;i++){
		fprintf(aux, "%5.4f\t", x_p[i]);
	}
	fprintf(aux,"\n");
	fclose(aux);
	aux=fopen("metrics.txt","a");
	fprintf(aux,"%10.2f\t%10.2f\t%10.2f\t",acctrai_svm,acctes_svm,accval_svm);
	fprintf(aux,"%10.2f\t%10.2f\t%10.2f\t",eqotrai_svm,eqotes_svm,eqoval_svm);
	fprintf(aux,"%10.2f\t%10.2f\t%10.2f\t",UNF_EqOpptrai_svm,UNF_EqOpptes_svm,UNF_EqOppval_svm);
	fprintf(aux,"%10.2f\t%10.2f\t%10.2f\t",UNF_PEtrai_svm,UNF_PEtes_svm,UNF_PEval_svm);
	fprintf(aux,"\n");
	fclose(aux);
	}

	else { // if sol is 108 or 106
	//Write the solution into file
	aux = fopen("databrooks.txt","a");
	printing_data_SVM (aux,  d, total_time, 99.0,solstat, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,x_p,0.0,acum_K, K_j);
	fclose(aux);
	aux = fopen("databrooks.txt","a");
	fprintf(aux,"\n");
	fclose(aux); 
	}

		TERMINATE:
		free_vectorint(label);
		free_vectorint(training_label);
		free_vectorint(validation_label);
		free_vectorchar(zctype);
		free_vector(x_p);
		free_matrix2(testing_sample,d);
		free_matrix2(training_sample,d);
		free_matrix2(validation_sample,d);
		free_vectorint(big_label);
		free_matrix2(big_sample,d);
		//Categorical
		free_vectorint(K_j);
		free_vectorint(acum_K);
		free_vectorint(cat_protected);
		//Fairness
		free_vectorint(v_eqop_tr);
		free_vectorint(v_eqop_test);
		free_vectorint(v_eqop_val);
		
	/* Free up the problem as allocated by CPXcreateprob, if necessary */

	if ( lp != NULL ) {
	status = CPXfreeprob (env, &lp);
	if ( status ) {
	fprintf (stderr, "CPXfreeprob failed, error code %d.\n", status);
	}
	}

	//	return 0; Iulia for 1 iteration
	}
	//close WHILE
	
	fclose(input);

	/* Free up the CPLEX environment, if necessary */

	if ( env != NULL ) {
	status = CPXcloseCPLEX (&env);

	/* Note that CPXcloseCPLEX produces no output,
		so the only way to see the cause of the error is to use
		CPXgeterrorstring.  For other CPLEX routines, the errors will
		be seen if the CPX_PARAM_SCRIND indicator is set to CPX_ON. */

	if ( status ) {
		char  errmsg[1024];
		fprintf (stderr, "Could not close CPLEX environment.\n");
		CPXgeterrorstring (env, status, errmsg);
		fprintf (stderr, "%s", errmsg);
	}
	}

	return(status);
	}/* END main */

 
 static int
setproblemdata_SVM (CPXENVptr *env, CPXLPptr *lp, int fila,int testsize,double Ci, double pi, int di,int M,double **sample,
	int *lab,double **tes_sample, int *tes_label, double **val_sample,int *val_label,double **x_p,double *objval,
	int *statusCR,int solve, double *total_time, int *g, double Delta, double beta){

	int status =0;
	time_t start, end;

	
	//Iulia: added
	double N=0.0;
	double P=0.0;
	///////////////////?
	
	int solstat,count,contador,h,qccount, qccontador, qch;
	int numcols= di+1+3*fila;  //Iulia: changed from int numcols= di+1+fila;
	int numrows= fila;
	int NUMNZ  = (di+2)*numrows;
	int NUMQNZ = di;
	int NUMCONQNZ= numcols*numcols; //should be fila for sums, fila*(di+1+3*fila) for 3rd, 4th ct

	double   acctes_cr,acctrai_cr,accval_cr;
	FILE * aux=NULL;

	if (solve==0){
		statusCR=0;

   char     *zprobname = 0;     /* Problem name <= 16 characters */
   double   *zobj = 0;
   double   *zrhs = 0;
   char     *zsense = 0;
   int      *zmatbeg = 0;
   int i,j;
   int      *zmatcnt = 0;
   int      *zmatind = 0;
   double   *zmatval = 0;
   double   *zlb = 0;
   double   *zub = 0;
   char     *zctype = 0;
   int      *zqmatbeg = 0;
   int      *zqmatcnt = 0;
   int      *zqmatind = 0;
   double   *zqmatval = 0;
   int *zqconrow = 0;
   int *zqconcol =0;
   double *zqconval=0;
   int      statusdata = 0;
   double **mat =0;

//////////////////////////////  Iulia:added for NEW quad constraints  QCPDUAL.C QCPEX1.C
	int *linnzcnt = 0; // need this...? # non0 linear coefficients in each quad constraint 
	int *quadnzcnt = 0; // # non0 quadratic coeff in quad constraint
	double *qrhs = 0;
	char *qsense  = 0;
	int **linind_matrix  = 0;
	double **linval_matrix  = 0;
	double **qcmat = 0; // cts linear terms

	linnzcnt = vectorint(fila+1); 
	quadnzcnt = vectorint(fila+1); // need this...? dual...
	qrhs = vector(2*fila+2);
	qsense  = vectorchar(2*fila+2); 
	linind_matrix = matrix2int(fila+1,numcols);
	linval_matrix  = matrix2(fila+1,numcols);
	qcmat = matrix2(fila+1,numcols);
/////////////////////////////////////////////////////////////////////////////
   mat = matrix2(numrows,numcols);
   zprobname = vectorchar(16);
   zobj      = vector(numcols);
   zrhs      = vector(numrows);
   zsense    = vectorchar(numrows);
   zmatbeg   = vectorint(numcols);
   zmatcnt   = vectorint(numcols);
   zmatind   = vectorint(NUMNZ);
   zmatval   = vector(NUMNZ);
   zlb       = vector(numcols);
   zub       = vector(numcols);
   zqmatbeg  = vectorint(numcols);
   zqmatcnt  = vectorint(numcols);
   zqmatind  = vectorint(NUMQNZ);
   zqmatval  = vector(NUMQNZ);
   zctype    =vectorchar(numcols);
   zqconrow  =vectorint(NUMCONQNZ);
   zqconcol  =vectorint(NUMCONQNZ);
   zqconval  =vector(NUMCONQNZ);

   if ( zprobname == NULL || zobj    == NULL ||
        zrhs      == NULL || zsense  == NULL ||
        zmatbeg   == NULL || zmatcnt == NULL ||
        zmatind   == NULL || zmatval == NULL ||
        zlb       == NULL || zub     == NULL ||
        zqmatbeg  == NULL || zqmatcnt == NULL ||
        zqmatind  == NULL || zqmatval == NULL ||
		zqconrow ==NULL || zqconcol == NULL ||
		zqconval == NULL  || linnzcnt == NULL || 
		quadnzcnt == NULL || qrhs == NULL || qsense == NULL  || linind_matrix == NULL ||
		linval_matrix == NULL )  {
      statusdata = 1;
      goto TERMINATE;
   }

   strcpy (zprobname, "svm");


   /* The code is formatted to make a visual correspondence
      between the mathematical linear program and the specific data
      items.   */


//Objective vector
  for (i=0;i<di+1;i++){ //w,b - 0 non0 coeff
	  zobj[i]=0.0; // ww/2 not here, not linear
   }

//OJOOOO  cambio C/n por C, para comparar con TABLE 1 brooks
    for (i = 0; i < fila; i++) {
		zobj[i+di+1]= (1-beta)*Ci/fila;
		zobj[i+di+1+fila]= beta; // Iulia: alpha_i
   }

   // !!!!


	//Escribimos la matriz de las restricciones:
		for (j=0;j<fila;j++){
		mat[j][di]=lab[j]; //+b
		for (i=0;i<di;i++){
			mat[j][i]=lab[j]*sample[i][j]; //Iulia: y_i * x_i for w_i
		}
	}

	for (i=0;i<fila;i++){
		for (j=0;j<fila;j++){
			if (i==j){
			mat[i][di+1+j]=1.0; // Iulia: +zeta_i (error)
			}
		}
	}
	
	 //Creamos los vectores beg ind cnt val
	 count=0;
	 contador=0;
	 h=0;
	 
	 for (j=0;j<numcols;j++){

		 for (i=0;i<numrows;i++){
			 
			 if (mat[i][j]!=0){
				 count=count+1;
				 if (count==1){ // first non-zero value on column j
					 zmatbeg[h]=contador;
					 h=h+1;
				 }
				 contador=contador+1;
			 }
		 }
	zmatcnt[j]=count;
	if (count==0){
		 zmatbeg[h]=zmatbeg[h-1];
		 h=h+1;
		 }
		 count=0;
	 }
	 count=0;
	 h=0;
	 for (j=0;j<numcols;j++){
		 for (i=0;i<numrows;i++){
			 if (mat[i][j]!=0){
				 count=count+1;
				 zmatval[h]=mat[i][j];
				 zmatind[h]=i;
				 h=h+1;
			 }
		 }
	 }

//Bound vectors

	 //ojooo
	for (i=0;i<di;i++){ //Julia: w
		 zlb[i]=-CPX_INFBOUND;
		 zub[i]=CPX_INFBOUND;
	 }

	zlb[di]=-CPX_INFBOUND; //Julia: b
	zub[di]= CPX_INFBOUND;
	 
   	for (i=0;i<fila;i++){ //Julia: zeta_i
	   zlb[di+1+i]=0.0;
	   zub[di+1+i]=CPX_INFBOUND;
  	 }

    for (i=0;i<fila;i++){ //Julia: alpha_i
	   zlb[di+1+fila+i]=0;
	   zub[di+1+fila+i]=1;
     }

    for (i=0;i<fila;i++){ //Julia: z_i
	   zlb[di+1+2*fila+i]=0;
	   zub[di+1+2*fila+i]=1;
     }

  /* The right-hand-side values don't fit nicely on a line above.  So put them here.  */
   for (i=0;i<fila;i++){ // lin *constraint*
   zsense[i] = 'G';
   zrhs[i]   = 1.0;
   }


//Iulia: added for quad constr

   qsense[0] = 'G';		 	qrhs[0]   = -Delta;
   qsense[1] = 'L';			qrhs[1]   =  Delta;

   for (i=2;i<fila+2;i++){ 
   qsense[i] = 'G';			qrhs[i]   =	 -M;
   qsense[fila+i] = 'L'; 	qrhs[fila+i] = 0.0;
   }

///////////////////////////////////////////

   /* Now set up the Q matrix.  Note that we set the values knowing that
    * we're doing a maximization problem, so negative values go on
    * the diagonal.  Also, the off diagonal terms are each repeated,
    * by taking the algebraic term and dividing by 2 */

   for (i=0;i<di;i++){
	   zqmatbeg[i]=i;
   }

   for (i=0;i<fila+1;i++){
	   zqmatbeg[di+i]=di;
   }

   for (i=0;i<di;i++){
	   zqmatcnt[i]=1;
   } 

    
//Iulia: changed for NEW constraints => + 2*FILA columns
/*  before 
	
	for (i=0;i<fila+1;i++){
  		 zqmatcnt[di+i] = 0;
   	}
*/

   for (i=0;i<3*fila+1;i++){
  		zqmatcnt[di+i] = 0;
   }

   for (i=0;i<di;i++){ // w_i * w_i ...?
	   zqmatind[i]=i;
	   zqmatval[i]=1-beta;
   }


////////////////////////////////////	Iulia: for NEW QC ~~~~~~~ linear terms
		for(i=0;i<fila+2*testsize;i++){
			if(g[i]>=1)
					P++;
			else
					N++;
		}

	//Quad Restriction matrix -> linear terms
	for (j=0;j<fila;j++){ //1st & 2nd constr for z_i
			qcmat[0][di+1+2*fila+j]=(1/P + 1/N)*g[j]-1/N; //CHECK row/col
	}

	for(i=1;i<fila+1;i++){
		qcmat[i][di] = 1.0; // b coeff
		for(j=0;j<di;j++){
			qcmat[i][j] = sample[j][i-1]; //x_i*w
		}

		for(j=0;j<fila;j++){
			if(i-1==j){ // -M*z_i
				qcmat[i][di+1+2*fila+j] = -M;
			}
		 }
	}

int cnt=0;
for(j=0;j<numcols;j++){ //for 1st,2nd constr. same terms
	if(qcmat[0][j]){ 
		linval_matrix[0][cnt] = qcmat[0][j];
		linind_matrix[0][cnt] = j; 
		cnt++;
	}
}
linnzcnt[0]=cnt;

for(i=1;i<fila+1;i++){
	cnt=0;
	for(j=0;j<numcols;j++){ //for 1st,2nd constr. same terms
		if(qcmat[i][j]){
			linval_matrix[i][cnt] = qcmat[i][j];
			linind_matrix[i][cnt] = j; 
			cnt++;
		}
	}
	linnzcnt[i]=cnt;
}
//////////////////////////////////////////////////////////////////////////////////////////
	/* Create the problem. */
   *lp = CPXcreateprob (*env, &status, zprobname);
      /* A returned pointer of NULL may mean that not enough memory
      was available or there was some other problem.  In the case of
      failure, an error message will have been written to the error
      channel from inside CPLEX.  In this example, the setting of
      the parameter CPX_PARAM_SCRIND causes the error message to
      appear on stdout.  */

   if ( *lp == NULL ) {
      fprintf (stderr, "Failed to create LP.\n");
      goto TERMINATE;
   }

   /* Now copy the problem data into the lp */

   status = CPXcopylp (*env, *lp, numcols, numrows, CPX_MIN, zobj, zrhs,
                       zsense, zmatbeg, zmatcnt, zmatind, zmatval,
                       zlb, zub, NULL);

   if ( status ) {
      fprintf (stderr, "Failed to copy problem data.\n");
      goto TERMINATE;
   }



 
	for (i=0;i<di+1;i++){
		zctype[i]='C';
	}

	for (i=0;i<fila;i++){
  	 	zctype[di+1+i] = 'C';
  	}

//Iulia: added for alpha_i and z_i
   	for (i=0;i<fila;i++){
  		zctype[di+1+fila+i] = 'I'; //alpha_i
   }
    for (i=0;i<fila;i++){
  		zctype[di+1+2*fila+i] = 'I'; //z_i 
   }  
////////////////////////////////////

	status = CPXcopyctype (*env, *lp, zctype);
   if ( status ) {
      fprintf (stderr, "Failed to copy ctype\n");
      goto TERMINATE;
   }


   status = CPXcopyquad (*env, *lp, zqmatbeg, zqmatcnt, zqmatind, zqmatval);
   if ( status ) {
      fprintf (stderr, "Failed to copy quadratic matrix.\n");
      goto TERMINATE;
   }

/* Now add the quadratic constraint. */

/* QCPEX1.C line 405+ */
/* int  CPXaddqconstr( CPXCENVptr env, CPXLPptr lp, int linnzcnt, int quadnzcnt, double rhs, int sense, int const * linind,
 double const * linval, int const * quadrow, int const * quadcol, double const * quadval, char const * lname_str ) */

// 1st,2nd constraints: Sums <= >= +-Delta 
	int qcnz=0;									 // W O R K 
	for (i=0;i<fila;i++){ 					    // skip w, b, zeta_i
		if(1/N-(1/P+1/N)*g[i]){ 
			 zqconrow[qcnz] = di+1+fila+i;    //row = alpha_i    
			 zqconcol[qcnz] = di+1+2*fila+i; //col = z_i
			 zqconval[qcnz] = 1/N-(1/P+1/N)*g[i];
			 qcnz++;
		}
	}

	quadnzcnt[0]=qcnz;

status = CPXaddqconstr (*env, *lp, linnzcnt[0], quadnzcnt[0], qrhs[0], qsense[0], linind_matrix[0], linval_matrix[0], zqconrow, zqconcol, zqconval, NULL);

   if ( status ) {
      fprintf (stderr, "Failed to copy first quadratic constraint.\n");
      goto TERMINATE;
    }

status = CPXaddqconstr (*env, *lp, linnzcnt[0], quadnzcnt[0], qrhs[1], qsense[1], linind_matrix[0], linval_matrix[0], zqconrow, zqconcol, zqconval, NULL);
   if ( status ) {
      fprintf (stderr, "Failed to copy second quadratic constraint.\n");
      goto TERMINATE;
    }

// 3rd,4th constraints
zqconrow  =vectorint(NUMCONQNZ);
zqconcol  =vectorint(NUMCONQNZ);
zqconval  =vector(NUMCONQNZ);

   for(int i=1;i<fila+1;i++){ //3rd constr
   	if(M){
	   	 	zqconrow[0] = di+1+fila+i-1;     
	    	zqconcol[0] = di+1+2*fila+i-1;
	    	zqconval[0] = M;
		}
	quadnzcnt[i]=1;
 	status = CPXaddqconstr (*env, *lp, linnzcnt[i], quadnzcnt[i], qrhs[2], qsense[2], linind_matrix[i], linval_matrix[i], zqconrow, zqconcol, zqconval, NULL);
	if ( status ) {
      fprintf (stderr, "Failed to copy (M) quadratic constraint.\n");
      goto TERMINATE;
    }
 	  status = CPXaddqconstr (*env, *lp, linnzcnt[i], quadnzcnt[i], qrhs[fila+2], qsense[fila+2], linind_matrix[i], linval_matrix[i], zqconrow, zqconcol, zqconval, NULL);
	  if ( status ) {
      fprintf (stderr, "Failed to copy (M) quadratic constraint.\n");
      goto TERMINATE;
      }
}

////////////////////////////////////////////////////////////////////////



/* Finally, write a copy of the problem to a file. */

   status = CPXwriteprob (*env, *lp, "CR.lp", NULL);
   if ( status ) {
      fprintf (stderr, "Failed to write LP to disk.\n");
      goto TERMINATE;
   }

     
      free_vectorchar(zprobname);
      free_vector(zobj);
      free_vector(zrhs);
      free_vectorchar(zsense);
      free_vectorint(zmatbeg);
      free_vectorint (zmatcnt);
      free_vectorint(zmatind);
      free_vector(zmatval);
      free_vector(zlb);
      free_vector(zub);
      free_vectorint(zqmatbeg);
      free_vectorint(zqmatcnt);
      free_vectorint(zqmatind);
      free_vector(zqmatval);
	  free_matrix2(mat,numrows);
	  free_vectorchar(zctype);
	  free_vectorint(zqconrow);
	  free_vectorint(zqconcol);
	  free_vector(zqconval);
   


}



else{

	   start = clock();

   /* Optimize the problem and obtain solution. */
   //Iulia: changed from CPXmipopt due to quadratic costraints. !! mipopt is also in main 
   status = CPXmipopt (*env, *lp);
   ////////////////////////////////////////////////////////////
   if ( status ) {
      fprintf (stderr, "Failed to optimize QP Brooks relaxed.\n");
      goto TERMINATE;
   }
     end = clock();

   *total_time = (double)( end - start )/(double)CLOCKS_PER_SEC ;


   solstat = CPXgetstat (*env, *lp);
   if (solstat!=101 && solstat!=102 && solstat!=105 && solstat!=107){
	   return solstat;
	   goto TERMINATE;
   }

   printf("\nSTATUS %d\n", solstat);
   *statusCR=solstat;
//
//      /* Write the output to the screen. */
//
   //printf ("\nSolution status = %d\n", solstat);

   status = CPXgetobjval (*env, *lp, objval);

   
   
   if ( status ) {
      fprintf (stderr,"No MIQP objective value available.  Exiting...\n");
      goto TERMINATE;
   }

   //printf ("Solution value  = %f\n\n", objval);

      /* The size of the problem should be obtained by asking CPLEX what
      the actual size is, rather than using what was passed to CPXcopylp.
      cur_numrows and cur_numcols store the current number of rows and
      columns, respectively.  */


   status = CPXgetx (*env, *lp, *x_p, 0, numcols-1);
   if ( status ) {
      fprintf (stderr, "Failed to get optimal integer x.\n");
      goto TERMINATE;
   }

   for (int j = 0; j < numcols; j++) {
      printf ( "Column %d:  Value = %17.10g\n", j, *x_p[j]);
   }

//Iulia: see other functions for NEW quad constraints (qcpex1.c)

   

   




accval_cr= Accuracy (testsize, di, *x_p, val_sample, val_label);
acctrai_cr= Accuracy (fila, di, *x_p, sample, lab);
acctes_cr= Accuracy (testsize, di, *x_p, tes_sample, tes_label);



TERMINATE:

status=status;

aux = fopen("relaxation.txt","a");
	printing_relaxation (aux, di, fila, *x_p);
	fclose(aux);



}

   return (status);

}  /* END setproblemdata_brooks */

 
 

/* This simple routine frees up the pointer *ptr, and sets *ptr to NULL */



static bool loadValues(FILE * file, int * FILA, double * C, int * a, double * p, int *d, int *M, int *delta2, double *beta,
					   double *mu1, double *sigma1, double *mu2, double *sigma2, int *outlier, double *tolerance,double *W,
					    int *nodelim, double *tilim){
	bool result = true;
	

	if (fscanf(file,"%d %lf %d %lf %d %d %d %lf %lf %lf %lf %lf %d %lf %lf %d %lf", FILA, C, a, p, d, M, delta2, beta, mu1, sigma1, mu2, sigma2, outlier,tolerance,W, nodelim, tilim) == EOF){
		result = false;
	}else{
	}

	return result;
}
//End of loadValues
