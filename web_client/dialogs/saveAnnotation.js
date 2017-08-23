import View from 'girder/views/View';

import saveAnnotation from '../templates/dialogs/saveAnnotation.pug';

/**
 * Create a modal dialog with fields to edit the properties of
 * an annotation before POSTing it to the server.
 */
var SaveAnnotation = View.extend({
    events: {
        'click .h-submit': 'save',
        'submit form': 'save'
    },

    render() {
        this.$el.html(
            saveAnnotation({
                annotation: this.annotation.toJSON()
            })
        ).girderModal(this);
        return this;
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

        this.annotation.set({
            name: this.$('#h-annotation-name').val(),
            description: this.$('#h-annotation-description').val()
        });
        this.annotation.trigger('g:save');
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
 * Show the save dialog box.  Watch for the `g:save` event on the
 * `AnnotationModel` to respond to user submission of the form.
 *
 * @param {AnnotationModel} annotationElement The element to edit
 * @returns {SaveAnnotation} The dialog's view
 */
function show(annotation) {
    dialog.annotation = annotation;
    dialog.setElement('#g-dialog-container').render();
    return dialog;
}

export default show;
