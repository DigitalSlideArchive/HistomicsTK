import BrowserWidget from 'girder/views/widgets/BrowserWidget';

import events from '../events';
import router from '../router';

var dialog;

function createDialog() {
    var widget = new BrowserWidget({
        parentView: null,
        titleText: 'Select a slide...',
        submitText: 'Open',
        showItems: true,
        selectItem: true,
        helpText: 'Click on a slide item to open.',
        rootSelectorSettings: {
            pageLimit: 50
        },
        validate: function (item) {
            if (!item.has('largeImage')) {
                return 'Please select a "large image" item.';
            }
        }
    });
    widget.on('g:saved', (model) => {
        if (!model) {
            return;
        }
        // reset image bounds when opening a new image
        router.setQuery('bounds', null, {trigger: false});
        router.setQuery('image', model.id, {trigger: true});
        $('.modal').girderModal('close');
    });
    return widget;
}

events.on('h:openImageUi', function () {
    if (!dialog) {
        dialog = createDialog();
    }
    dialog.setElement($('#g-dialog-container')).render();
});
