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
"trace"         { return TRACE; }
"rank"          { return RANK; }
"lu"            { return LU; }
"qr"            { return QR; }

"q"             { exit(0); }
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
     printf("\nBasic Functions:\n");
     printf("- detA          : Calculate determinant of matrix A\n");
     printf("- transpose A   : Get transpose of matrix A\n");
     printf("- inverse A     : Calculate inverse of matrix A\n");
     printf("- trace A       : Sum of diagonal elements of A\n");
     printf("- rank A        : Calculate rank of matrix A\n");
     printf("- eigenval A    : Calculate eigenvalues of A\n");
     printf("\nDecompositions:\n");
     printf("- lu A          : LU decomposition (returns [L;U])\n");
     printf("- qr A          : QR decomposition (returns [Q;R])\n");
     
     
     printf("\nEnter expression: ");
     input();         /* Consume \n */
     return HELP;     /* Return HELP token */
 }

{WHITESPACE}    { /* Ignore whitespace */ }
{NEWLINE}       { 
                  return '\n';
                }

.               { /* Unknown character */
                  return ERROR_TOKEN;  /* Return custom error token */
                }
%%

int yywrap() 
{
    return 1; 
}
