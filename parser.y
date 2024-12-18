%{
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "parser.tab.h"

/* External declarations for lexical analyzer */
extern int yylex(void);
void yyerror(const char *s);

/* Flag to prevent multiple error messages for the same syntax error */
int syntax_error_printed = 0;

/* Matrix operation error handler
 * Handles errors from matrix operations where the result is NULL
 * The specific error message is printed by matrix_perror
 */
void handle_matrix_error(const char* op, Matrix* result) {
    if (!result) {
        /* Error message already printed by matrix_perror */
    }
}
%}

/* Define data types for semantic values
 * These types are used to store values during parsing
 */
%union {
    double num;           /* For numeric literals */
    Matrix* matrix;       /* For matrix operations */
}

/* Operator precedence and associativity (lowest to highest)
 * This defines the order of operations for the calculator
 */
%left PLUS MINUS                   /* Addition and subtraction - lowest priority */
%left MULTIPLY DIVIDE              /* Multiplication and division */
%right POWER                       /* Exponentiation - highest priority */
%nonassoc DET INVERSE TRANSPOSE    /* Function calls */
%nonassoc EIGENVAL EIGENVEC       
%nonassoc LU QR                   
%nonassoc TRACE RANK              
%nonassoc EXP LOG NORM            
%nonassoc '(' ')'                 /* Parentheses */

/* Token definitions for lexical analysis */
%token <num> NUMBER                              /* Numeric values */
%token LBRACKET RBRACKET COMMA SEMICOLON        /* Matrix delimiters */
%token PLUS MINUS MULTIPLY DIVIDE POWER          /* Basic operators */
%token DET TRANSPOSE INVERSE EIGENVAL EIGENVEC   /* Matrix operations */
%token LU QR TRACE RANK                         /* Advanced matrix operations */
%token ERROR_TOKEN                              /* For error handling */
%token HELP                                    /* Help command */
%token EXP LOG NORM

/* Type declarations for non-terminal symbols */
%type <matrix> matrix expr
%type <matrix> matrix_rows matrix_row

%%

/* Grammar Rules Section */

/* Main program structure
 * Handles input processing and error recovery
 */
program: lines
       ;

lines: lines expr '\n'   { 
        if ($2) {
            printf("Answer:\n");
            print_matrix($2); 
            free_matrix($2);
        }
        printf("\nEnter expression: ");
        fflush(stdout);
        syntax_error_printed = 0;
     }
     | lines error '\n' { 
        yyerrok;
        if (!syntax_error_printed) {
            printf("Error: Syntax error\n");
            syntax_error_printed = 1;
        }
        printf("\nEnter expression: ");
        fflush(stdout);
        syntax_error_printed = 0;
     }
     | lines ERROR_TOKEN '\n' {
        yyerrok;
        if (!syntax_error_printed) {
            printf("Error: Syntax error\n");
            syntax_error_printed = 1;
        }
        printf("\nEnter expression: ");
        fflush(stdout);
        syntax_error_printed = 0;
     }
     | lines HELP      { /* 不做任何操作，继续等待新的输入 */ }
     | /* empty */     { printf("Enter expression: "); fflush(stdout); }
     ;

/* Expression Rules
 * Define all possible matrix operations and their syntax
 */
expr: matrix                { $$ = $1; }
    | '(' expr ')'         { $$ = $2; }
    | expr POWER NUMBER    { 
        Matrix* exp = create_matrix_element($3);
        $$ = matrix_power($1, exp);
        free_matrix(exp);
    }
    | expr MULTIPLY expr   { $$ = matrix_multiply($1, $3); }
    | expr DIVIDE expr     { $$ = matrix_divide($1, $3); }
    | expr PLUS expr       { $$ = matrix_add($1, $3); }
    | expr MINUS expr      { $$ = matrix_subtract($1, $3); }
    | DET expr             { $$ = matrix_determinant($2); }
    | TRANSPOSE expr       { $$ = matrix_transpose($2); }
    | INVERSE expr         { $$ = matrix_inverse($2); }
    | EIGENVAL expr        { $$ = matrix_eigenvalues($2); }
    | EIGENVEC expr        { $$ = matrix_eigenvectors($2); }
    | LU expr              { $$ = matrix_lu_decomposition($2); }
    | QR expr              { $$ = matrix_qr_decomposition($2); }
    | EXP expr             { $$ = matrix_exp($2); }
    | LOG expr             { $$ = matrix_log($2); }
    | NORM expr            {
        double norm_val = matrix_norm($2);
        $$ = create_matrix_element(norm_val);
    }
    | TRACE expr           {
        double trace = matrix_trace($2);
        $$ = create_matrix_element(trace);
    }
    | RANK expr            {
        double rank_val = matrix_rank($2);
        $$ = create_matrix_element(rank_val);
    }
    ;

/* Matrix Construction Rules
 * Define syntax for creating matrices
 */
matrix: LBRACKET matrix_rows RBRACKET    { $$ = $2; }        /* Matrix [1,2;3,4] */
       ;

/* Matrix construction */
matrix_rows: matrix_row                          { $$ = $1; }
           | matrix_rows SEMICOLON matrix_row     { $$ = append_row($1, $3); }
           ;

matrix_row: NUMBER                               { $$ = create_matrix_element($1); }
          | matrix_row COMMA NUMBER              { $$ = add_element($1, $3); }
          ;

%%

/* Error handling function
 * Prints syntax error message only once per error occurrence
 */
void yyerror(const char *s) 
{
    if (!syntax_error_printed) {
        printf("Error: Syntax error\n");
        syntax_error_printed = 1;
    }
} 
