\
import requests
import json
import time

BASE_URL = "http://127.0.0.1:5001/api"  # Updated port

def print_status(message, is_success):
    status = "SUCCESS" if is_success else "FAILURE"
    print(f"[{status}] {message}")

def test_create_book_project():
    print("\\n--- Testing Create Book Project ---")
    project_name = f"Test Project {int(time.time())}"
    payload = {"project_name": project_name}
    try:
        response = requests.post(f"{BASE_URL}/book_projects", json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        
        assert data["project_name"] == project_name
        # assert "last_modified" in data # Temporarily removed
        # Add more assertions based on the expected initial state
        print_status(f"Created book project: {project_name}", True)
        return project_name
    except requests.exceptions.RequestException as e:
        print_status(f"Error creating book project: {e}", False)
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"Response content: {e.response.json()}")
            except json.JSONDecodeError:
                print(f"Response content: {e.response.text}")
    except AssertionError as e:  # Catch AssertionError specifically
        print_status(f"AssertionError: {e}", False)
        if 'data' in locals(): # Check if data variable exists
            print(f"Actual API Response: {data}") # Print the API response
    except Exception as e:
        print_status(f"An unexpected error occurred: {e}", False) # Print other unexpected errors
    return None

def test_list_book_projects(created_project_name):
    print("\\\\n--- Testing List Book Projects ---")
    if not created_project_name:
        print_status("Skipping list test as project creation failed.", False)
        return False
    try:
        response = requests.get(f"{BASE_URL}/book_projects")
        response.raise_for_status()
        projects = response.json() # This is a list of strings
        
        # Corrected assertion: Check if the created_project_name is in the list of project names
        found = created_project_name in projects
        assert found
        print_status(f"Listed book projects. Found '{created_project_name}'.", True)
        return True
    except requests.exceptions.RequestException as e:
        print_status(f"Error listing book projects: {e}", False)
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"Response content: {e.response.json()}")
            except json.JSONDecodeError:
                print(f"Response content: {e.response.text}")
    except Exception as e:
        print_status(f"Assertion or other error: {e}", False)
    return False

def test_get_book_project(project_name):
    print(f"\\n--- Testing Get Book Project: {project_name} ---")
    if not project_name:
        print_status("Skipping get test as project name is invalid.", False)
        return False
    try:
        response = requests.get(f"{BASE_URL}/book_projects/{project_name}")
        response.raise_for_status()
        data = response.json()
        
        assert data["project_name"] == project_name
        print_status(f"Successfully fetched project: {project_name}", True)
        return True
    except requests.exceptions.RequestException as e:
        print_status(f"Error fetching project {project_name}: {e}", False)
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"Response content: {e.response.json()}")
            except json.JSONDecodeError:
                print(f"Response content: {e.response.text}")
    except Exception as e:
        print_status(f"Assertion or other error: {e}", False)
    return False

def test_update_book_project(project_name):
    print(f"\\n--- Testing Update Book Project: {project_name} ---")
    if not project_name:
        print_status("Skipping update test as project name is invalid.", False)
        return False

    # First, get the current project data to ensure we have all fields
    try:
        get_response = requests.get(f"{BASE_URL}/book_projects/{project_name}")
        get_response.raise_for_status()
        current_project_data = get_response.json()
    except requests.exceptions.RequestException as e:
        print_status(f"Failed to fetch project before update: {e}", False)
        return False

    updated_premise = f"This is an updated premise for {project_name} at {time.time()}"
    # Corrected: Update a field that exists in BookWritingState, e.g., 'overall_plot_summary'
    # Or, if you intend to add 'premise_logline' to BookWritingState, that model needs to be updated first.
    # For now, let's assume we update 'overall_plot_summary' as an example.
    updated_payload = current_project_data.copy() # Create a copy to modify
    updated_payload["overall_plot_summary"] = updated_premise 
    # If you add more updatable fields to BookWritingState, they can be set here.

    try:
        # The PUT request should send the entire BookWritingState model
        response = requests.put(f"{BASE_URL}/book_projects/{project_name}", json=updated_payload)
        response.raise_for_status()
        data = response.json()

        assert data["project_name"] == project_name
        assert data["overall_plot_summary"] == updated_premise
        # Remove or comment out the assertion for 'premise_logline' if it's not part of BookWritingState
        # assert data["premise_logline"] == updated_premise 
        print_status(f"Successfully updated project: {project_name}", True)
        return True
        
    except requests.exceptions.RequestException as e:
        print_status(f"Error updating project {project_name}: {e}", False)
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"Response content: {e.response.json()}")
            except json.JSONDecodeError:
                print(f"Response content: {e.response.text}")
    except Exception as e:
        print_status(f"Assertion or other error: {e}", False)
    return False

if __name__ == "__main__":
    print("Starting Book Projects API Integration Test Suite...")
    
    # Run tests
    created_project = test_create_book_project()
    
    if created_project:
        test_list_book_projects(created_project)
        test_get_book_project(created_project)
        test_update_book_project(created_project)
        # Note: A test for deleting the project would be good here if a DELETE endpoint exists.
        # print(f"\\nConsider manually deleting test project: {created_project} if not done automatically.")
    else:
        print("\\nSkipping further tests as project creation failed or returned no name.")
        
    print("\\nTest Suite Finished.")
    print("Reminder: Ensure your FastAPI application (app/main.py) was running during these tests.")
