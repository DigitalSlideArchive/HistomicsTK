import _ from 'underscore';

import View from 'girder/views/View';

import saveAnnotation from '../templates/dialogs/saveAnnotation.pug';

/**
 * Create a modal dialog with fields to edit the properties of
 * an annotation before POSTing it to the server.
 */
var SaveAnnotation = View.extend({
    events: {
        'click .h-cancel': 'cancel',
        'submit form': 'save'
    },

    render() {
        this.$el.html(
            saveAnnotation({
                title: this.options.title,
                annotation: this.annotation.toJSON().annotation
            })
        ).girderModal(this);
        return this;
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
