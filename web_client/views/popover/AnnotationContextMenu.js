import _ from 'underscore';

import StyleCollection from '../../collections/StyleCollection';
import View from '../View';

import template from '../../templates/popover/annotationContextMenu.pug';
import '../../stylesheets/popover/annotationContextMenu.styl';

const AnnotationContextMenu = View.extend({
    events: {
        'click .h-remove-elements': '_removeElements',
        'click .h-set-group': '_setGroup',
        'click .h-remove-group': '_removeGroup'
    },
    initialize(settings) {
        this.reset();
    },
    render() {
        const hovered = this._hovered || {};
        const currentGroup = (hovered.element || {}).group;
        this.$el.html(template({
            groups: this._getAnnotationGroups(),
            currentGroup
        }));
        return this;
    },
    reset() {
        if (!this._hovered) {
            return;
        }
        this.parentView.trigger('h:highlightAnnotation');
        this._hovered = null;
    },
    setHovered(element, annotation) {
        const elementModel = annotation.elements().get(element.id);
        if (annotation._pageElements || !elementModel) {
            // ignore context menu actions on paged annotations
            return;
        }
        this.reset();
        this._hovered = { element, annotation };
        this.render();
        window.setTimeout(() => {
            this.parentView.trigger('h:highlightAnnotation', annotation.id, element.id);
        }, 1);
    },
    _removeElements(evt) {
        evt.preventDefault();
        evt.stopPropagation();

        const { annotation, element } = this._hovered;
        annotation.elements().remove(element);
        this.reset();
        this.trigger('h:close');
    },
    _setStyleDefinition(group) {
        const styles = new StyleCollection();
        return styles.fetch().done(() => {
            const style = styles.get({ id: group || 'default' });
            const { element } = this._hovered;
            const elementModel = this._hovered.annotation.elements().get(element);
            const styleAttrs = Object.assign({}, style.toJSON());
            delete styleAttrs.id;
            if (group) {
                styleAttrs.group = group;
            }
            elementModel.set(styleAttrs);
            if (!group) {
                elementModel.unset('group');
            }
        }).always(() => {
            this.reset();
            this.trigger('h:close');
        });
    },
    _getAnnotationGroups() {
        const groups = _.union(...this.collection.map((a) => a.get('groups')));
        groups.sort();
        return groups.slice(0, 10);
    },
    _setGroup(evt) {
        evt.preventDefault();
        evt.stopPropagation();

        const group = $(evt.currentTarget).data('group');
        this._setStyleDefinition(group);
    },
    _removeGroup(evt) {
        evt.preventDefault();
        evt.stopPropagation();

        this._setStyleDefinition(null);
    }
});

export default AnnotationContextMenu;
