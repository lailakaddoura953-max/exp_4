"""
Camera-Location-Aware Hazard Rules engine.

This package resolves a camera's physical location type from its Ocularis
display name (or, for training data, its folder name), looks up the
location-specific hazard rules, and applies those rules to detections.

Modules:
    camera_location_resolver.py        - parses live Ocularis camera names
    training_folder_location_resolver.py - parses image_data_with_synth/ folder names
    rules.py                           - defines what the rules are (data + loader)
    check_rules_from_object_label.py   - checks a single detection against a rule set
    orchestrator.py                    - ties it together end-to-end per camera
    audit_logger.py                    - JSON-lines audit trail for rule decisions
"""
