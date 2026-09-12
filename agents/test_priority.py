#For checking just one specific drug, here's the quick way to see if it's a priority drug or not
from priority_candidates import get_priority_candidates

priority = get_priority_candidates()  # uses the default relative path
drug_name = "KETAMINE"
print(drug_name.strip().lower() in priority)