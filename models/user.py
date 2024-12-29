user_schema = {
    'auth0_id': str,  # Auth0 user ID
    'email': str,
    'name': str,
    'picture': str,  # Profile picture URL
    'last_login': datetime,
    'created_at': datetime,
    'analyses': [  # List of analysis IDs
        {
            'analysis_id': ObjectId,
            'created_at': datetime,
            'type': str  # document, image, or video
        }
    ]
} 