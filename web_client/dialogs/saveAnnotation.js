import _ from 'underscore';
import tinycolor from 'tinycolor2';

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
        // clean up old colorpickers when rerendering
        this.$('.h-colorpicker').colorpicker('destroy');

        const showStyleEditor = this.annotation.get('annotation').elements && !this.annotation._pageElements;
        const defaultStyles = {};

        if (showStyleEditor) {
            const elements = this.annotation.get('annotation').elements;
            console.assert(elements.length > 0); // otherwise we wouldn't show the style editor
            const firstElement = elements[0];
            if (elements.every((d) => d.lineWidth === firstElement.lineWidth)) {
                defaultStyles.lineWidth = firstElement.lineWidth;
            }
            if (elements.every((d) => d.lineColor === firstElement.lineColor)) {
                defaultStyles.lineColor = firstElement.lineColor;
            }
            if (elements.every((d) => d.fillColor === firstElement.fillColor)) {
                defaultStyles.fillColor = firstElement.fillColor;
            }
        }

        this.$el.html(
            saveAnnotation({
                title: this.options.title,
                hasAdmin: this.annotation.get('_accessLevel') >= AccessType.ADMIN,
                annotation: this.annotation.toJSON().annotation,
                showStyleEditor,
                defaultStyles
            })
        ).girderModal(this);
        this.$('.h-colorpicker').colorpicker();
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

        let validation = '';

        if (!this.$('#h-annotation-name').val()) {
            this.$('#h-annotation-name').parent().addClass('has-error');
            validation += 'Please enter a name. ';
        }

        const setFillColor = !!this.$('#h-annotation-fill-color').val();
        const fillColor = tinycolor(this.$('#h-annotation-fill-color').val()).toRgbString();
        const setLineColor = !!this.$('#h-annotation-line-color').val();
        const lineColor = tinycolor(this.$('#h-annotation-line-color').val()).toRgbString();
        const setLineWidth = !!this.$('#h-annotation-line-width').val();
        const lineWidth = parseFloat(this.$('#h-annotation-line-width').val());

        if (setLineWidth && (lineWidth < 0 || !isFinite(lineWidth))) {
            validation += 'Invalid line width. ';
            this.$('#h-annotation-line-width').parent().addClass('has-error');
        }

        if (validation) {
            this.$('.g-validation-failed-message').text(validation.trim())
                .removeClass('hidden');
            return;
        }

        // all valid

        if (setFillColor || setLineColor || setLineWidth) {
            this.annotation.elements().each((element) => { /* eslint-disable backbone/no-silent */
                if (setFillColor) {
                    element.set('fillColor', fillColor, {silent: true});
                }
                if (setLineColor) {
                    element.set('lineColor', lineColor, {silent: true});
                }
                if (setLineWidth) {
                    element.set('lineWidth', lineWidth, {silent: true});
                }
            });
            const annotationData = _.extend({}, this.annotation.get('annotation'));
            annotationData.elements = this.annotation.elements().toJSON();
            this.annotation.set('annotation', annotationData);
        }

        _.extend(this.annotation.get('annotation'), {
            name: this.$('#h-annotation-name').val(),
            description: this.$('#h-annotation-description').val()
        });
        this.trigger('g:submit');
        this.$el.modal('hide');
    },

    destroy() {
        this.$('.h-colorpicker').colorpicker('destroy');
        SaveAnnotation.prototype.destroy.call(this);
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
    _.defaults(options, { title: 'Create annotation' });
    dialog.annotation = annotation;
    dialog.options = options;
    dialog.setElement('#g-dialog-container').render();
    return dialog;
}

export default show;
