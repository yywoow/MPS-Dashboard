#!/usr/bin/env python3

# Script to make UI changes to app.py

# Read the original file
with open('app_backup.py', 'r') as f:
    content = f.read()

# Make the changes
content = content.replace('st.title("MPS Dashboard Tool")', 'st.title("MPS Playground")')

# Remove the MAPPING_NOTE lines
lines = content.split('\n')
new_lines = []
for line in lines:
    if 'MAPPING_NOTE = ' in line or 'st.caption(MAPPING_NOTE)' in line:
        continue
    new_lines.append(line)

content = '\n'.join(new_lines)

# Add CSS for condensed UI after the title
css_code = '''
# CSS for more condensed UI
st.markdown(
	"""
	<style>
	/* Reduce general padding and margins */
	.main .block-container {
		padding-top: 1rem;
		padding-bottom: 1rem;
	}
	
	/* Reduce spacing between elements */
	.stMarkdown, .stSelectbox, .stMultiSelect, .stRadio {
		margin-bottom: 0.5rem !important;
	}
	
	/* Compact form styling */
	.stForm {
		border: 1px solid #e0e0e0;
		padding: 1rem;
		margin: 0.5rem 0;
	}
	
	/* Reduce button spacing */
	.stButton > button {
		margin: 0.2rem 0;
	}
	
	/* Compact subheaders */
	.stSubheader {
		margin-top: 1rem;
		margin-bottom: 0.5rem;
	}
	
	/* Reduce tab content padding */
	.stTabs [data-baseweb="tab-panel"] {
		padding-top: 1rem;
	}
	</style>
	""",
	unsafe_allow_html=True,
)
'''

# Insert CSS after title
content = content.replace(
    'st.title("MPS Playground")', 
    'st.title("MPS Playground")\n' + css_code
)

# Write the updated content
with open('app.py', 'w') as f:
    f.write(content)

print("UI changes applied successfully!")
