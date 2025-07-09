# file: hakgal_grammar.py

HAKGAL_GRAMMAR = r"""
    ?start: formula

    formula: expression "."

    ?expression: quantified_formula
               | implication

    ?implication: disjunction ( "->" implication )?

    ?disjunction: conjunction ( "|" disjunction )?

    ?conjunction: negation ( "&" conjunction )?

    ?negation: "-" atom_expression
             | atom_expression
    
    ?atom_expression: atom
                    | "(" expression ")"

    quantified_formula: "all" VAR "(" expression ")"

    atom: PREDICATE ("(" [arg_list] ")")?
    
    arg_list: term ("," term)*
    
    ?term: PREDICATE | VAR

    // MODIFIED: Added the hyphen '-' to the character set for subsequent characters.
    PREDICATE: /[A-ZÄÖÜ][a-zA-ZÄÖÜäöüß0-9_-]*/
    VAR: /[a-z][a-zA-Z0-9_]*/

    %import common.WS
    %ignore WS
"""