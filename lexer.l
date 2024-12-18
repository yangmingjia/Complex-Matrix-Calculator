%{
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "parser.tab.h"
%}

/* Regular definitions */
DIGIT       [0-9]
INTEGER     {DIGIT}+
FLOAT       {INTEGER}\.{INTEGER}?|\.{INTEGER}
NUMBER      {INTEGER}|{FLOAT}
WHITESPACE  [ \t\r]+
NEWLINE     \n

%%
{NUMBER}        { 
                  yylval.num = atof(yytext); 
                  return NUMBER; 
                }

"["             { return LBRACKET; }
"]"             { return RBRACKET; }
","             { return COMMA; }
";"             { return SEMICOLON; }
"("             { return '('; }
")"             { return ')'; }

"+"             { return PLUS; }
"-"             { return MINUS; }
"*"             { return MULTIPLY; }
"/"             { return DIVIDE; }
"^"             { return POWER; }

"det"           { return DET; }
"transpose"     { return TRANSPOSE; }
"inverse"       { return INVERSE; }
"eigenval"      { return EIGENVAL; }
"eigenvec"      { return EIGENVEC; }
"trace"         { return TRACE; }
"rank"          { return RANK; }
"lu"            { return LU; }
"qr"            { return QR; }
"exp"           { return EXP; }
"log"           { return LOG; }
"norm"          { return NORM; }
"quit"          { exit(0); }
"help"          { 
     printf("\nMatrix Calculator Help\n");
     printf("===================\n");
     printf("\nOperations:\n");
     printf("- A + B         : Matrix addition\n");
     printf("- A - B         : Matrix subtraction\n");
     printf("- A * B         : Matrix multiplication\n");
     printf("- A / B         : Matrix division (A * B^(-1))\n");
     printf("- A ^ n         : Matrix power (integer n >= 0)\n");
     printf("Basic Functions:\n");
     printf("- detA          : Calculate determinant of matrix A\n");
     printf("- transpose A   : Get transpose of matrix A\n");
     printf("- inverse A     : Calculate inverse of matrix A\n");
     printf("- trace A       : Sum of diagonal elements of A\n");
     printf("- rank A        : Calculate rank of matrix A\n");
     printf("- norm A        : Calculate Frobenius norm of A\n");
     printf("\nMatrix Functions:\n");
     printf("- exp A         : Matrix exponential e^A\n");
     printf("- log A         : Matrix natural logarithm\n");
     printf("\nDecompositions:\n");
     printf("- lu A          : LU decomposition (returns [L;U])\n");
     printf("- qr A          : QR decomposition (returns [Q;R])\n");
     printf("\nEigenvalue Analysis:\n");
     printf("- eigenval A    : Calculate eigenvalues of A\n");
     printf("- eigenvec A    : Calculate eigenvectors of A\n");
     
     
     printf("\nEnter expression: ");
     input();         /* 消耗掉换行符 */
     return HELP;     /* 返回HELP token */
 }

{WHITESPACE}    { /* ignore whitespace */ }
{NEWLINE}       { 
                  return '\n';
                }

.               { /* unknown character */
                  return ERROR_TOKEN;  /* Return custom error token */
                }
%%

int yywrap() 
{
    return 1;
}

