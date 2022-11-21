#!/bin/bash

# switch to terminator
# gnome-terminal -- sh -c "source entrypoint.sh && cd diagnosis_va && rasa train --domain domain.yl --data data --out models"

gnome-terminal -- sh -c "source entrypoint.sh && cd diagnosis_va/actions && rasa run actions"

gnome-terminal -- sh -c "source entrypoint.sh && cd diagnosis_va && rasa shell"
