%{
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "parser.tab.h"
extern int yylex(void);
void yyerror(const char *s);
%}

%union {
    double num;
    Matrix* matrix;
    Complex complex;
}

/* 定义运算符优先级和结合性 */
%left PLUS MINUS
%left MULTIPLY DIVIDE
%right POWER
%left TRANSPOSE
%left DET EIGENVAL EIGENVEC LU QR SCHUR TRACE RANK INVERSE

%token <num> NUMBER
%token LBRACKET RBRACKET COMMA SEMICOLON
%token PLUS MINUS MULTIPLY DIVIDE POWER
%token DET TRANSPOSE INVERSE EIGENVAL EIGENVEC
%token LU QR SCHUR TRACE RANK

%type <matrix> matrix
%type <matrix> matrix_rows
%type <matrix> matrix_row
%type <matrix> expr
%type <complex> complex_number

%%

program: program expr '\n'   { print_matrix($2); free_matrix($2); }
        | /* empty */        { }
        ;

expr: matrix                { $$ = $1; }
    | expr PLUS expr        { $$ = matrix_add($1, $3); }
    | expr MINUS expr       { $$ = matrix_subtract($1, $3); }
    | expr MULTIPLY expr    { $$ = matrix_multiply($1, $3); }
    | expr DIVIDE expr      { $$ = matrix_divide($1, $3); }
    | expr POWER expr       { $$ = matrix_power($1, $3); }
    | DET expr             { $$ = matrix_determinant($2); }
    | TRANSPOSE expr       { $$ = matrix_transpose($2); }
    | INVERSE expr         { $$ = matrix_inverse($2); }
    | EIGENVAL expr        { $$ = matrix_eigenvalues($2); }
    | EIGENVEC expr        { $$ = matrix_eigenvectors($2); }
    | LU expr              { $$ = matrix_lu_decomposition($2); }
    | QR expr              { $$ = matrix_qr_decomposition($2); }
    | SCHUR expr           { $$ = matrix_schur_decomposition($2); }
    | TRACE expr           { 
        Complex trace = matrix_trace($2);
        $$ = create_matrix_element(trace);
    }
    | RANK expr            { 
        Complex rank_val = {(double)matrix_rank($2), 0};
        $$ = create_matrix_element(rank_val);
    }
    | '(' expr ')'         { $$ = $2; }
    ;

matrix: LBRACKET matrix_rows RBRACKET    { $$ = $2; }
      ;

matrix_rows: matrix_row                          { $$ = $1; }
          | matrix_rows SEMICOLON matrix_row     { $$ = append_row($1, $3); }
          ;

matrix_row: complex_number                       { $$ = create_matrix_element($1); }
         | matrix_row COMMA complex_number       { $$ = add_element($1, $3); }
         ;

complex_number: NUMBER                           { $$ = complex_from_real($1); }
              | '(' NUMBER ',' NUMBER ')'        { 
                                                    Complex c = {$2, $4};
                                                    $$ = c;
                                                }
              ;

%%

void yyerror(const char *s) {
    fprintf(stderr, "Error: %s\n", s);
} 
