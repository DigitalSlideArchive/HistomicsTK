import StyleCollection from '../../collections/StyleCollection';
import View from '../View';

import template from '../../templates/popover/annotationContextMenu.pug';
import '../../stylesheets/popover/annotationContextMenu.styl';

const AnnotationContextMenu = View.extend({
    events: {
        'click .h-remove-elements': '_removeElements',
        'click .h-edit-elements': '_editElements',
        'click .h-set-group': '_setGroup',
        'click .h-remove-group': '_removeGroup'
    },
    initialize(settings) {
        this.styles = new StyleCollection();
        this.styles.fetch().done(() => this.render());
        this.listenTo(this.collection, 'add remove reset', this.render);
    },
    render() {
        this.$el.html(template({
            groups: this._getAnnotationGroups(),
            numberSelected: this.collection.length
        }));
        return this;
    },
    _removeElements(evt) {
        evt.preventDefault();
        evt.stopPropagation();

        this.collection.trigger('h:remove');
        this.trigger('h:close');
    },
    _editElements(evt) {
        evt.preventDefault();
        evt.stopPropagation();

        this.trigger('h:edit', this.collection.at(0));
        this.trigger('h:close');
    },
    _setStyleDefinition(group) {
        const style = this.styles.get({ id: group }) || this.styles.get({ id: 'default' });
        const styleAttrs = Object.assign({}, style.toJSON());
        delete styleAttrs.id;
        this.collection.each((element) => { /* eslint-disable backbone/no-silent */
            if (group) {
                styleAttrs.group = group;
            } else {
                element.unset('group', {silent: true});
            }
            element.set(styleAttrs, {silent: true});
            // test
            JSON.stringify(element.toJSON());
        });
        this.collection.trigger('h:save');
        this.trigger('h:close');
    },
    _getAnnotationGroups() {
        const groups = this.styles.map((style) => style.id);
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
