# hakgal_grammar.py (Version 4.1 - Unicode Support)
# Fügt Unterstützung für deutsche Umlaute und Eszett in Prädikatnamen hinzu.

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

    PREDICATE: /[A-ZÄÖÜ][a-zA-ZÄÖÜäöüß0-9_]*/
    VAR: /[a-z][a-zA-Z0-9_]*/

    %import common.WS
    %ignore WS
"""