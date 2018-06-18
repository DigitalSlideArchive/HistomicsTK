import _ from 'underscore';

import AccessWidget from 'girder/views/widgets/AccessWidget';
import View from 'girder/views/View';
import { AccessType } from 'girder/constants';

import saveAnnotation from '../templates/dialogs/saveAnnotation.pug';
import '../stylesheets/dialogs/saveAnnotation.styl';

/**
 * Create a modal dialog with fields to edit the properties of
 * an annotation before POSTing it to the server.
 */
var SaveAnnotation = View.extend({
    events: {
        'click .h-access': 'access',
        'click .h-cancel': 'cancel',
        'submit form': 'save'
    },

    render() {
        this.$el.html(
            saveAnnotation({
                title: this.options.title,
                hasAdmin: this.annotation.get('_accessLevel') >= AccessType.ADMIN,
                annotation: this.annotation.toJSON().annotation
            })
        ).girderModal(this);
        return this;
    },

    access(evt) {
        evt.preventDefault();
        this.annotation.off('g:accessListSaved');
        new AccessWidget({
            el: $('#g-dialog-container'),
            type: 'annotation',
            hideRecurseOption: true,
            parentView: this,
            model: this.annotation
        }).on('g:accessListSaved', () => {
            this.annotation.fetch();
        });
    },

    cancel(evt) {
        evt.preventDefault();
        this.$el.modal('hide');
    },

    /**
     * Respond to form submission.  Triggers a `g:save` event on the
     * AnnotationModel.
     */
    save(evt) {
        evt.preventDefault();
        if (!this.$('#h-annotation-name').val()) {
            this.$('#h-annotation-name').parent()
                .addClass('has-error');
            this.$('.g-validation-failed-message')
                .text('Please enter a name.')
                .removeClass('hidden');
            return;
        }

        _.extend(this.annotation.get('annotation'), {
            name: this.$('#h-annotation-name').val(),
            description: this.$('#h-annotation-description').val()
        });
        this.trigger('g:submit');
        this.$el.modal('hide');
    }
});

/**
 * Create a singleton instance of this widget that will be rendered
 * when `show` is called.
 */
var dialog = new SaveAnnotation({
    parentView: null
});

/**
 * Show the save dialog box.  Watch for the `g:submit` event on the
 * view to respond to user submission of the form.
 *
 * @param {AnnotationModel} annotationElement The element to edit
 * @returns {SaveAnnotation} The dialog's view
 */
function show(annotation, options) {
    _.defaults(options, {'title': 'Create annotation'});
    dialog.annotation = annotation;
    dialog.options = options;
    dialog.setElement('#g-dialog-container').render();
    return dialog;
}

export default show;
