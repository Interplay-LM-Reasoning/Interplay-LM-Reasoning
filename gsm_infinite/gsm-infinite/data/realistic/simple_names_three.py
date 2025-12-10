hierarchical_categorizations = [
["Location", "total number of adult animals", "total number of newborn animal children"], 
] 

subcategories = {
    "Location": { 
        "Wild place": ["Beverly Forest", "Cedar Valley", "Oakridge Riverside", "Pine Ridge", "Maple Creek"], 
        "Human place": ["South Zoo", "Jefferson Circus", "Mayer Aquarium", "Bundle Ranch", "Hamilton Farm"], 
    }, 
    "total number of adult animals": { 
        "Mammal": ["adult bear", "adult wolf", "adult deer", "adult fox", "adult racoon"],
        "Bird": ["adult eagle", "adult parrot", "adult owl", "adult blue jay", "adult crow"], 
    }, 
    "total number of newborn animal children": {
        "number": ["average number of newborn children"], 
    }, 
} 

messagetwo = "Answer the questions below. \
Note: the total number of adult animals in one location refers to sum of all types of adult animals ever mentioned for the specific location throughout the problem EXCLUDING their number of newborn children. \
Important: If the a type of animal is never mentioned for a location, assume its inexistence in that location. \
Each question is INDEPENDENT of the others." 

message = "Answer the questions below. \
Note: the total number of adult animals in one location refers to sum of all types of adult animals ever mentioned for the specific location throughout the problem EXCLUDING their number of newborn children. \
IMPORTANT: if the a type of animal is never mentioned for a location for that specific problem, assume its INEXISTENCE in that location (number in the location = 0). Previous problems animal quantities are not applicable for the present one. \
The average number of newborn children of the same type of animal might vary across different locations. \
The total newborn animal children of a location refers to the sum of the TOTAL newborn children (not average newborn children) from all adult animals mentioned for that specific location. \
Hint: the total newborn children of one type of animal in a location equals the average children per that animal in location times the number of that animal in location. \
Each question is self-contained INDEPENDENT of the others. Quantities of previous problems is NOT correct for the new problem, so quantities of animals MUST be recalculated for each question! \
Final answer needs to be presented in the format 'Answer: xxx', where xxx is the number you calculated." 
lengthmessage = 40 

hierarchical_categorizationstwo = [
    ["City", "total number of schools", "total number of teachers from all schools"], 
] 

subcategoriestwo = { 
    "City": {
        "Names without water": ["Evervale City", "Hawkesbury", "Glenfield City", "Westhaven City", "Brightford"], 
        "Names with water": ["Riverton City", "Clearwater Bay", "Shoreline City", "Oakbridge City", "Ruby Bay"], 
    }, 
    "total number of schools": {
        # "Common": ["Elementary School", "Public Highschool", "Private Christian Highschool", "Private Middle School", "Technical College"], 
        # "Specialty": ["Acting School", "Regional Medical School", "Culinarian School", "Regional Law School"], 
        "Common": ["elementary school", "private middle school", "public highschool", "regional medical school", "culinarian school"], 
    }, 
    "total number of teachers from all schools": {
        "number": ["average number of teachers"], 
    }, 
} 
messagetwo = "Answer the questions below. \
Note: the total number of schools in one location refers to sum of all types of schools ever mentioned for the specific location throughout the problem. \
IMPORTANT: if the a type of school is never mentioned for a location for that specific problem, assume its INEXISTENCE in that location (number in the location = 0). Previous problems school quantities are not applicable for the present one. \
The average number of teachers of the same type of school might vary across different locations. \
The number of teachers from all schools of a location refers to the sum of the TOTAL teachers (not average number of teachers) from all type of schools mentioned for that specific location. \
Hint: the number of teachers of one type of school in a location equals the average teacher per that type of schools in location times the number of that type of schools in location. \
Each question is self-contained INDEPENDENT of the others. Quantities of previous problems is NOT correct for the new problem, so quantities of schools or teachers MUST be recalculated for each question! \
Final answer needs to be presented in the format 'Answer: xxx', where xxx is the number you calculated." 
lengthmessagetwo = 40 

hierarchical_categorizationsthree = [
    ["Festivals", "total number of movies", "total number of movie nominations"], 
] 

subcategoriesthree = {
    "Festivals": {
        # "Visionary": ["Morgan Glass", "Luca Verdi", "Emma Steele", "Ravi Kapoor", "Sasha Liu"], 
        "Random": ["Taylor Movie Festival", "Verdi Movie Festival", "West Sahara Movie Festival", "Northwood Movie Festival", "Golden Banana Movie Festival"],
        # "Comedy": ["Mia Lark", "Oscar Bloom", "Nina Wilde", "Felix Grant", "Dana Sparks"], 
        "French": ["Festival Lumière de Valmont", "Rêves de Belleville", "Cinéma de Montreval", "Festival de Clairmont", "Festival de Saint-Rivage"], 
    }, 
    "total number of movies": {
        # "Drama": ["Modern Family Drama", "Solemn Period Drama", "Futuristic Sci-Fi Movie", "Mythical Adventure Movie", "Realistic Detective Thriller"], 
        # "Crime": ["Urban Love Comedy", "Gothic Horror Movie", "Upbeat Fantasy Musical", "Intense Sports Drama", "Calm Road Movie"], 
        "Common": ["upbeat metropolis comedy", "solemn period drama", "futuristic sci-fi movie", "calm road movie", "intense detective thriller"], 
    }, 
    "total number of movie nominations": {
        "number": ["average number of nominations"], 
    } 
} 
messagethree = "Answer the questions below. \
Commedy, drama, and thriller are all movies types. \
Note: the total number of movies in one movie festival refers to sum of all types of movies ever mentioned for the specific movie festival throughout the problem. \
IMPORTANT: if the a type of movies is never mentioned for a location for that specific problem, assume its INEXISTENCE in that location (number in the location = 0). Previous problems movies quantities are not applicable for the present one. \
The average number of nominations of the same type of festival might vary across different locations. \
The number of total movie nominations from all movies of a festival refers to the sum of the TOTAL movie nominations (not average number of nominations) from all types of movies mentioned for that specific location. \
Hint: the number of total movie nomination of one type of movie in a festival equals the average nomination per that type of movie in festival times the number of that type of movies in festival. \
Each question is self-contained INDEPENDENT of the others. Quantities of previous problems is NOT correct for the new problem, so quantities of movies or nominations MUST be recalculated for each question! \
Final answer needs to be presented in the format 'Answer: xxx', where xxx is the number you calculated." 
lengthmessagethree = 55 
