%{
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "parser.tab.h"

int line_number = 1;
int yywrap(void);
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

{WHITESPACE}    { /* ignore whitespace */ }
{NEWLINE}       { 
                  line_number++;
                  printf("\nEnter expression: ");
                  fflush(stdout);
                  return '\n';
                }

.               { 
                  printf("Error at line %d: Unknown character '%s'\n", 
                         line_number, yytext);
                  return -1;
                }
%%

int yywrap() {
    return 1;
}

