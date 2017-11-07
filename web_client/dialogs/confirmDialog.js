import $ from 'jquery';
import _ from 'underscore';

import View from 'girder/views/View';
import 'girder/utilities/jquery/girderModal';

import events from '../events';

import template from '../templates/dialogs/confirmDialog.pug';
// import '../stylesheets/dialogs/openAnnotatedImage.styl';

let dialog = null;
const defaultOptions = {
    title: 'Warning',
    message: 'Are you sure?',
    submitButton: 'Yes',
    onSubmit: _.noop

};

const ConfirmDialog = View.extend({
    events: {
        'click .h-submit': '_submit'
    },
    render() {
        this.$el.html(template(this._options)).girderModal(this);
        return this;
    },

    setOptions(options) {
        this._options = _.defaults(options, defaultOptions);
    },

    _submit() {
        this.trigger('h:submit', this._options);
        this.$el.modal('hide');
        this.off('h:submit');
    }
});

events.on('h:confirmDialog', function (options = {}) {
    if (!dialog) {
        dialog = new ConfirmDialog({
            parentView: null
        });
    }
    dialog.off('h:submit');
    dialog.setOptions(options);
    dialog.on('h:submit', options.onSubmit);
    dialog.setElement($('#g-dialog-container')).render();
});
